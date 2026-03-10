"""Stem separation API."""
import asyncio
import io
import json
import logging
import sys
import uuid
import zipfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

jobs: dict[str, dict] = {}

app = FastAPI(
    title="Stem separation API",
    description="Upload audio; get 4 stems (vocals, drums, bass, other) or vocals + accompaniment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_STEMS = BACKEND_ROOT / "checkpoints" / "spectrogram_stems" / "best.pt"
CHECKPOINT_VOCALS = BACKEND_ROOT / "checkpoints" / "spectrogram" / "best.pt"
CHUNK_SECONDS = 6.0

_estimator_stems = None
_estimator_vocals = None


def get_estimator_stems():
    global _estimator_stems
    if _estimator_stems is None:
        if not CHECKPOINT_STEMS.exists():
            raise HTTPException(status_code=503, detail=f"Checkpoint not found: {CHECKPOINT_STEMS}")
        from src.inference.separate import build_spectrogram_stems_estimator
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _estimator_stems = build_spectrogram_stems_estimator(CHECKPOINT_STEMS, chunk_seconds=CHUNK_SECONDS, device=device)
    return _estimator_stems


def get_estimator_vocals():
    global _estimator_vocals
    if _estimator_vocals is None:
        if not CHECKPOINT_VOCALS.exists():
            raise HTTPException(status_code=503, detail=f"Checkpoint not found: {CHECKPOINT_VOCALS}")
        from src.inference.separate import build_spectrogram_estimator
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _estimator_vocals = build_spectrogram_estimator(CHECKPOINT_VOCALS, chunk_seconds=CHUNK_SECONDS, device=device)
    return _estimator_vocals


class Track:
    def __init__(self, audio: np.ndarray, rate: int):
        self.audio = audio
        self.rate = rate


def _write_stems_zip(stems: dict, rate: int, filename_stem: str, zip_name: str) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in stems.items():
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 1:
                data = data[:, np.newaxis]
            wav_buf = io.BytesIO()
            sf.write(wav_buf, data, rate, format="WAV")
            zf.writestr(f"{name}.wav", wav_buf.getvalue())
    buf.seek(0)
    return buf


def _run_separation(job_id: str, contents: bytes, filename: str, model: str) -> None:
    try:
        jobs[job_id]["message"] = "Loading audio..."
        audio, rate = sf.read(io.BytesIO(contents), dtype="float32")
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        track = Track(audio, rate)
        stem_name = Path(filename or "audio").stem

        def on_progress(processed: int, total: int) -> None:
            pct = int(100 * processed / total) if total else 0
            jobs[job_id]["progress"] = min(pct, 99)
            jobs[job_id]["message"] = f"Separating... {pct}%"

        if model == "stems":
            estimator = get_estimator_stems()
            stems = estimator(track, progress_callback=on_progress)
            buf = _write_stems_zip(stems, rate, stem_name, f"{stem_name}_stems.zip")
            zip_filename = f"{stem_name}_stems.zip"
        else:
            estimator = get_estimator_vocals()
            stems = estimator(track, progress_callback=on_progress)
            buf = _write_stems_zip(stems, rate, stem_name, f"{stem_name}_vocals.zip")
            zip_filename = f"{stem_name}_vocals.zip"

        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Building ZIP..."
        jobs[job_id]["result"] = buf.read()
        jobs[job_id]["filename"] = zip_filename
        logger.info("job %s completed: %s", job_id, zip_filename)
    except Exception as e:
        logger.exception("job %s failed", job_id)
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = -1


def _api_info():
    return {
        "message": "Stem separation API. POST /separate with an audio file.",
        "models": [
            {
                "id": "stems",
                "name": "Stems",
                "description": "4-stem separation: vocals, drums, bass, other.",
            },
            {
                "id": "vocals",
                "name": "Vocals",
                "description": "Vocals and accompaniment only.",
            },
        ],
    }


@app.get("/")
def root():
    return _api_info()


@app.get("/separate/info")
def separate_info():
    """Models info at /separate/info so nginx can route it (root / goes to frontend)."""
    return _api_info()


@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    model: str = Query("stems", description="stems (4-stem) or vocals (vocals + accompaniment)"),
):
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Upload a WAV, MP3, FLAC, OGG, or M4A file")
    if model not in ("stems", "vocals"):
        raise HTTPException(status_code=400, detail="model must be 'stems' or 'vocals'")

    contents = await file.read()
    try:
        audio, rate = sf.read(io.BytesIO(contents), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    track = Track(audio, rate)
    stem_name = Path(file.filename or "audio").stem

    if model == "stems":
        estimator = get_estimator_stems()
        stems = estimator(track)
        buf = _write_stems_zip(stems, rate, stem_name, f"{stem_name}_stems.zip")
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={stem_name}_stems.zip"},
        )
    else:
        estimator = get_estimator_vocals()
        stems = estimator(track)
        buf = _write_stems_zip(stems, rate, stem_name, f"{stem_name}_vocals.zip")
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={stem_name}_vocals.zip"},
        )


@app.post("/separate/start")
async def separate_start(
    file: UploadFile = File(...),
    model: str = Query("stems", description="stems or vocals"),
):
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Upload a WAV, MP3, FLAC, OGG, or M4A file")
    if model not in ("stems", "vocals"):
        raise HTTPException(status_code=400, detail="model must be 'stems' or 'vocals'")

    contents = await file.read()
    try:
        audio, rate = sf.read(io.BytesIO(contents), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"progress": 0, "message": "Starting...", "result": None, "error": None, "filename": None}
    asyncio.create_task(asyncio.to_thread(_run_separation, job_id, contents, file.filename or "audio", model))
    return {"job_id": job_id}


async def _progress_stream(job_id: str):
    last = -1
    while True:
        if job_id not in jobs:
            yield f"data: {json.dumps({'progress': -1, 'error': 'Job not found'})}\n\n"
            return
        j = jobs[job_id]
        progress = j.get("progress", 0)
        message = j.get("message", "")
        err = j.get("error")
        if err:
            yield f"data: {json.dumps({'progress': -1, 'error': err})}\n\n"
            return
        if progress >= 100:
            yield f"data: {json.dumps({'progress': 100, 'message': 'Done', 'done': True})}\n\n"
            return
        if progress != last:
            last = progress
            yield f"data: {json.dumps({'progress': progress, 'message': message})}\n\n"
        await asyncio.sleep(0.3)


@app.get("/separate/progress/{job_id}")
async def separate_progress(job_id: str):
    return StreamingResponse(
        _progress_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/separate/result/{job_id}")
async def separate_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    if j.get("error"):
        raise HTTPException(status_code=500, detail=j["error"])
    if j.get("result") is None:
        return JSONResponse(status_code=202, content={"status": "processing"})
    filename = j.get("filename") or "stems.zip"
    return StreamingResponse(
        io.BytesIO(j["result"]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
