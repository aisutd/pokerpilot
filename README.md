# PokerPilot

## Description

PokerPilot is an AI-powered poker decision-making platform that learns how individual opponents play and adapts its strategy in real time.

The system combines behavioral data, game-state signals, and machine learning to recommend profitable actions and explain its reasoning in plain language.

## Planned Technologies

- Python
- TypeScript
- SQL
- React / Next.js
- FastAPI
- Machine learning and data analysis tools

## Docker Setup

Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

| Service | What it runs | URL |
|---|---|---|
| `backend` | FastAPI + XGBoost model (`backend/`, `pokerpilot_model/`) | http://localhost:8000 (Swagger at `/docs`) |
| `frontend` | `frontend/index.html` served by nginx | http://localhost:3000 |
| `trainer` | `pokerpilot_model/train.py` (on demand only) | — |

Optional: copy `backend/.env.example` to `backend/.env` and add `OPENAI_API_KEY` for real GPT replies. Without it, `/api/pilot` returns a mock reply.

```bash
docker compose up -d --build   # build and start backend + frontend
docker compose ps              # status (backend shows "healthy" once the model loads)
docker compose logs -f backend # follow logs
docker compose down            # stop everything
```

Retrain the model (about 6 minutes; writes new files into `pokerpilot_model/artifacts/`, then rebuild the backend to pick them up):

```bash
docker compose --profile train run --rm trainer
docker compose up -d --build backend
```
