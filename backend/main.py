
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import sys
from pathlib import Path
ssys.path.append(str(Path(__file__).resolve().parent.parent / "pokerpilot_model"))
from predict import PokerPilotModel

load_dotenv()  

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(
    title="Poker Pilot API",
    description="Backend for the Poker Pilot project",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)
model = PokerPilotModel()   # loads once when the server starts


@app.post("/api/recommend")
def recommend(state: dict):
    return model.recommend(state)


class UserInput(BaseModel):
    username: str


class PilotResponse(BaseModel):
    username: str
    message: str
    source: str  



def ask_gpt(prompt: str) -> str:
    """Small wrapper around the OpenAI API. Swap the prompt/model here."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    result = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are Poker Pilot, a friendly poker coach. Keep answers under 80 words."},
            {"role": "user", "content": prompt},
        ],
    )
    return result.choices[0].message.content


@app.get("/")
def root():
    return {"message": "Poker Pilot API is running. Go to /docs for Swagger UI."}


@app.get("/health")
def health():
    """API HEALTH check - used by Docker and to confirm the server is up."""
    return {"status": "ok", "gpt_configured": bool(OPENAI_API_KEY)}


@app.post("/api/pilot", response_model=PilotResponse)
def pilot(data: UserInput):
    """Takes a username and returns output based on it."""
    name = data.username.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Username can't be empty.")

    
    if not OPENAI_API_KEY:
        return PilotResponse(
            username=name,
            message=f"Welcome to the table, {name}! (Mock reply: add OPENAI_API_KEY to get real GPT output.)",
            source="mock",
        )

    try:
        reply = ask_gpt(f"Greet the player '{name}' and give them one beginner poker tip.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GPT request failed: {e}")

    return PilotResponse(username=name, message=reply, source="gpt")
