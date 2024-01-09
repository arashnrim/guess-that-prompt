import io
import os
import sqlite3
import uuid
import base64

import google.generativeai as genai
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from dotenv import load_dotenv
from flask import Flask, send_file
from PIL import Image
import asyncio

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("Please set the GOOGLE_API_KEY environment variable.")
    exit()

app = Flask(__name__)
queue = asyncio.Queue()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configures the Stable Diffusion model to generate an image based on the prompt
model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Configures Google's Gemini Pro model
model = genai.GenerativeModel("gemini-pro")


with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, prompt TEXT, timeTaken INTEGER)")


@app.route("/rounds/start", methods=["POST"])
async def start_round():
    round_id = generate_round_id()
    print(f"{round_id}: Request received")

    if torch.cuda.is_available():
        try:
            # Calls on Google's Gemini Pro model to generate a prompt for the image-to-text model
            response = model.generate_content(
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
    else:
        print("CUDA is not available.")


@app.route("/rounds/<round_id>/status", methods=["GET"])
def round_status(round_id):
    if round_id is None:
        return {"error": "No round ID provided"}, 400

    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        result = cursor.execute(
            "SELECT * FROM games WHERE id=?", (round_id,)).fetchone()

        if result is None:
            return {"id": round_id, "status": "pending", "message": "Generation is still ongoing..."}, 202

        return {"id": round_id, "status": "done", "message": "Generation is complete. Call the `getRound` endpoint to get the image."}, 200


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
