
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "pokerpilot_model"))
from predict import PokerPilotModel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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



def ask_gemini(prompt: str) -> str:
    """Small wrapper around the Gemini API."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    result = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            "You are Poker Pilot, a friendly poker coach. "
            "Keep answers under 80 words. " + prompt
        ),
    )
    return result.text


@app.get("/")
def root():
    return {"message": "Poker Pilot API is running. Go to /docs for Swagger UI."}


@app.get("/health")
def health():
    """API HEALTH check - used by Docker and to confirm the server is up."""
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}


@app.post("/api/pilot", response_model=PilotResponse)
def pilot(data: UserInput):
    """Takes a username and returns output based on it."""
    name = data.username.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Username can't be empty.")

    
    if not GEMINI_API_KEY:
        return PilotResponse(
            username=name,
            message=f"Welcome to the table, {name}! (Mock reply: add GEMINI_API_KEY to get real Gemini output.)",
            source="mock",
        )

    try:
        reply = ask_gemini(f"Greet the player '{name}' and give them one beginner poker tip.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {e}")

    return PilotResponse(username=name, message=reply, source="gemini")
