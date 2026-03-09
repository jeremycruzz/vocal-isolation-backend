# Stem separation backend

Minimal backend to run stem separation on uploaded audio: either **4-stem** (vocals, drums, bass, other) or **vocals-only** (vocals + accompaniment). Extracted from the main `cap` repo so it can be deployed or moved to a separate repo.

## Setup

1. **Create a virtualenv and install dependencies:**

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Add checkpoints:**

   Copy the trained model(s) from the parent repo:

   ```bash
   # 4-stem (required for model=stems)
   mkdir -p checkpoints/spectrogram_stems
   cp ../outputs/spectrogram_stems/best.pt checkpoints/spectrogram_stems/

   # Vocals-only (required for model=vocals)
   mkdir -p checkpoints/spectrogram
   cp ../outputs/spectrogram/best.pt checkpoints/spectrogram/
   ```

## Run the API

From the `backend` directory (so `src` and `app` are on the path):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **GET /** — API info and available models
- **POST /separate** — upload an audio file (WAV, MP3, FLAC, OGG, M4A)
  - **?model=stems** (default) — returns a zip with `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`
  - **?model=vocals** — returns a zip with `vocals.wav`, `accompaniment.wav`

Examples:

```bash
# 4-stem separation (default)
curl -X POST -F "file=@song.mp3" "http://localhost:8000/separate" -o stems.zip

# Vocals-only separation
curl -X POST -F "file=@song.mp3" "http://localhost:8000/separate?model=vocals" -o vocals.zip
```

## Docker

```bash
docker compose up --build
```

- **API:** http://localhost:8000
- **Frontend:** http://localhost (requires `cap-frontend` as sibling directory)

For frontend live reload (watch), use the dev override. Use **Docker Compose V2** (plugin: `docker compose` with a space):
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```
Frontend: http://localhost:5173 (Vite HMR). Source under `../cap-frontend` is mounted into the container.

If you only have the legacy `docker-compose` (v1, with a hyphen), run `down` first to avoid a ContainerConfig error, then up:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```
To get Compose V2: [Install the Docker Compose plugin](https://docs.docker.com/compose/install/).

## CI/CD

Builds and pushes to ECR, then deploys to EC2 on push to `main`. See [.github/DEPLOY.md](.github/DEPLOY.md) for required secrets and variables.

API: http://localhost:8000

For AWS deployment (EC2, ECS, App Runner), see the Dockerfile and ensure checkpoints are available (volume mount or baked into the image).

## Layout

- `app/main.py` — FastAPI app (upload → separate → zip response)
- `src/models/spectrogram/` — model code (STFT, U-Net, vocals + stems)
- `src/inference/separate.py` — load models, chunked separation, estimators
- `checkpoints/` — `spectrogram_stems/best.pt` and/or `spectrogram/best.pt`
