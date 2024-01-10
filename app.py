import asyncio
import base64
import io
import os
import sqlite3
import uuid

import google.generativeai as genai
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from dotenv import load_dotenv
from flask import Flask

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("Please set the GOOGLE_API_KEY environment variable.")
    exit()

# Initializes the Flask app
app = Flask(__name__)

# Initializes the Gemini Pro model for use later
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel("gemini-pro")

# Initializes the Stable Diffusion model for use later
scheduler = EulerDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2", subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2", scheduler=scheduler, torch_dtype=torch.float16)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    print("Warning: CUDA is not available. The model will run on the CPU. This will be slower.")

# Connects to and creates the database if it doesn't exist
with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, prompt TEXT, timeTaken INTEGER)")


@app.route("/rounds/start", methods=["POST"])
async def start_round():
    """Begins a new round of the game. This endpoint is synchronous and will block until the image is generated."""
    round_id = generate_round_id()
    print(f"{round_id}: Request received")

    try:
        # Calls on Google's Gemini Pro model to generate a prompt for the image-to-text model
        response = gemini.generate_content(
            "Generate a prompt for an image-to-text model by ONLY giving a single sentence in the following format: \"A/an <adjective> <subject> <adverb> <verb> a/an <adjective> <subject>.\" Use complicated verbs and actions to get the generated image to harder to guess.")
        print(f"{round_id}: Text-to-image prompt generated: {response.text}")

        # Generates the image and saves it
        if not os.path.exists("images/"):
            os.mkdir("images/")

        # Generates and saves the image
        prompt = response.text
        image = pipe(prompt).images[0]
        image.save(f"images/{round_id}.png")
        print(f"{round_id}: Image generated and saved to images/{round_id}.png")

        # Saves the prompt to the database
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO games (id, prompt) VALUES (?, ?)", (round_id, prompt))
            conn.commit()

        # Converts the image to base64 so that the client can display it
        with open(f"images/{round_id}.png", "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        # Returns the data to the client
        return {"id": round_id, "prompt": prompt, "image": encoded_image}, 200
    except Exception as e:
        print(f"{round_id}: Error occurred: {e}")
        return {"error": "An error occurred while generating the image."}, 500


def generate_round_id() -> str:
    """
    Generates a random ID for a round. The ID is ensured to be unique.

    Returns:
        `str`: The generated ID.
    """
    round_id = str(uuid.uuid4())
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        if cursor.execute("SELECT * FROM games WHERE id=?", (round_id,)).fetchone() is not None:
            return generate_round_id()
    return round_id
