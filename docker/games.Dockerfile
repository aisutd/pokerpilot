# One image for both PokerKit game apps (single-player on 5000, multiplayer on
# 5001). docker-compose.yml picks which app.py each container runs.
# Build context is the repo root: the apps import backend/, pokerpilot_model/
# and each other through paths relative to the repo root.

# --- Chat UIs (Vite builds served by the Flask apps at /chat/) ---
FROM node:22-alpine AS chat-ui
WORKDIR /build

COPY client/package.json client/package-lock.json client/
RUN npm ci --prefix client
COPY Multiplayer-Mode/client/package.json Multiplayer-Mode/client/package-lock.json Multiplayer-Mode/client/
RUN npm ci --prefix Multiplayer-Mode/client

COPY client/ client/
COPY Multiplayer-Mode/client/ Multiplayer-Mode/client/
RUN npm run build --prefix client \
    && npm run build --prefix Multiplayer-Mode/client

# --- Python apps ---
FROM python:3.12-slim

# xgboost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY PokerPilot-Website/requirements.txt PokerPilot-Website/requirements.txt
COPY Multiplayer-Mode/requirements.txt Multiplayer-Mode/requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir \
    -r PokerPilot-Website/requirements.txt \
    -r Multiplayer-Mode/requirements.txt \
    -r backend/requirements.txt

COPY backend/ backend/
COPY pokerpilot_model/ pokerpilot_model/
COPY PokerPilot-Website/ PokerPilot-Website/
COPY Multiplayer-Mode/ Multiplayer-Mode/
# The folder is "Assets" in git but may be "assets" on case-insensitive disks.
COPY [Aa]ssets/ Assets/

COPY --from=chat-ui /build/client/dist client/dist
COPY --from=chat-ui /build/Multiplayer-Mode/client/dist Multiplayer-Mode/client/dist

ENV HOST=0.0.0.0
EXPOSE 5000 5001
CMD ["python", "PokerPilot-Website/app.py"]
