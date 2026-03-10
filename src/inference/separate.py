"""Load spectrogram models and separate: vocals-only (vocals + accompaniment) or 4-stem (vocals, drums, bass, other)."""
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)
import torch

from src.models.spectrogram import SpectrogramStemsSeparator, SpectrogramVocalSeparator, STEM_NAMES


def load_spectrogram_model(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> SpectrogramVocalSeparator:
    """Load vocals-only spectrogram model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = SpectrogramVocalSeparator(
        n_fft=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512),
        win_length=config.get("win_length"),
        unet_base=config.get("unet_base", 32),
        unet_depth=config.get("unet_depth", 4),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        activation=config.get("activation", "leaky_relu"),
        mask_type=config.get("mask_type", "alpha"),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def separate_track_chunked(
    model: Any,
    mix_audio: np.ndarray,
    chunk_seconds: float = 6.0,
    overlap_ratio: float = 0.5,
    sample_rate: int = 44100,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Chunked separation for vocals-only model; returns vocals waveform."""
    if device is None:
        device = next(model.parameters()).device
    if mix_audio.ndim == 1:
        mix_audio = mix_audio[:, np.newaxis]
    chunk_samples = int(chunk_seconds * sample_rate)
    hop = int(chunk_samples * (1 - overlap_ratio))
    if hop < 1:
        hop = 1
    n_samples, n_channels = mix_audio.shape
    out = np.zeros_like(mix_audio, dtype=np.float32)
    weight = np.zeros((n_samples, 1), dtype=np.float32)
    start = 0
    chunk_idx = 0
    with torch.no_grad():
        while start < n_samples:
            end = min(start + chunk_samples, n_samples)
            if end - start < 100:
                break
            chunk = mix_audio[start:end]
            if chunk.shape[0] < chunk_samples:
                chunk = np.pad(
                    chunk,
                    ((0, chunk_samples - chunk.shape[0]), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            x = torch.from_numpy(chunk.T[np.newaxis, :, :]).float().to(device)
            v = model(x, length=chunk.shape[0])
            v = v[0].cpu().numpy().T
            want = end - start
            got = v.shape[0]
            n = min(want, got)
            out[start : start + n] += v[:n]
            weight[start : start + n] += 1.0
            start += hop
            chunk_idx += 1
            if progress_callback:
                progress_callback(start, n_samples)
            log.info("vocals chunk %s: %s / %s samples", chunk_idx, start, n_samples)
            if start >= n_samples:
                break
    weight[weight < 1e-6] = 1.0
    out /= weight
    return out


def build_spectrogram_estimator(
    checkpoint_path: str | Path,
    chunk_seconds: float = 6.0,
    device: Optional[torch.device] = None,
) -> Any:
    """Build estimator that returns dict: vocals, accompaniment."""
    model = load_spectrogram_model(checkpoint_path, device)
    if device is None:
        device = next(model.parameters()).device

    def estimator(track, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, np.ndarray]:
        mix = np.asarray(track.audio, dtype=np.float32)
        if mix.ndim == 1:
            mix = mix[:, np.newaxis]
        rate = int(track.rate)
        vocals = separate_track_chunked(
            model, mix, chunk_seconds=chunk_seconds, overlap_ratio=0.5, sample_rate=rate, device=device,
            progress_callback=progress_callback,
        )
        accompaniment = mix - vocals
        return {"vocals": vocals, "accompaniment": accompaniment}

    return estimator


def load_spectrogram_stems_model(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> SpectrogramStemsSeparator:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = SpectrogramStemsSeparator(
        n_fft=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512),
        win_length=config.get("win_length"),
        unet_base=config.get("unet_base", 32),
        unet_depth=config.get("unet_depth", 4),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        n_sources=config.get("n_sources", 4),
        activation=config.get("activation", "leaky_relu"),
        mask_type=config.get("mask_type", "alpha"),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def separate_track_chunked_stems(
    model: Any,
    mix_audio: np.ndarray,
    chunk_seconds: float = 6.0,
    overlap_ratio: float = 0.5,
    sample_rate: int = 44100,
    device: Optional[torch.device] = None,
    stem_names: Optional[list] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    if mix_audio.ndim == 1:
        mix_audio = mix_audio[:, np.newaxis]
    stem_names = stem_names or getattr(model, "n_sources", 4)
    if isinstance(stem_names, int):
        stem_names = list(STEM_NAMES)[:stem_names]
    chunk_samples = int(chunk_seconds * sample_rate)
    hop = int(chunk_samples * (1 - overlap_ratio))
    if hop < 1:
        hop = 1
    n_samples, n_channels = mix_audio.shape
    out: Dict[str, np.ndarray] = {name: np.zeros_like(mix_audio, dtype=np.float32) for name in stem_names}
    weight = np.zeros((n_samples, 1), dtype=np.float32)

    start = 0
    chunk_idx = 0
    with torch.no_grad():
        while start < n_samples:
            end = min(start + chunk_samples, n_samples)
            if end - start < 100:
                break
            chunk = mix_audio[start:end]
            if chunk.shape[0] < chunk_samples:
                chunk = np.pad(
                    chunk,
                    ((0, chunk_samples - chunk.shape[0]), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            x = torch.from_numpy(chunk.T[np.newaxis, :, :]).float().to(device)
            stems = model(x, length=chunk.shape[0])
            want = end - start
            for name in stem_names:
                v = stems[name][0].cpu().numpy().T
                got = v.shape[0]
                n = min(want, got)
                out[name][start : start + n] += v[:n]
            weight[start : start + n] += 1.0
            start += hop
            chunk_idx += 1
            if progress_callback:
                progress_callback(start, n_samples)
            log.info("stems chunk %s: %s / %s samples", chunk_idx, start, n_samples)
            if start >= n_samples:
                break

    weight[weight < 1e-6] = 1.0
    for name in stem_names:
        out[name] = out[name] / weight
    return out


def build_spectrogram_stems_estimator(
    checkpoint_path: str | Path,
    chunk_seconds: float = 6.0,
    device: Optional[torch.device] = None,
) -> Any:
    model = load_spectrogram_stems_model(checkpoint_path, device)
    if device is None:
        device = next(model.parameters()).device
    stem_names = STEM_NAMES[: model.n_sources]

    def estimator(track, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, np.ndarray]:
        mix = np.asarray(track.audio, dtype=np.float32)
        if mix.ndim == 1:
            mix = mix[:, np.newaxis]
        rate = int(track.rate)
        return separate_track_chunked_stems(
            model,
            mix,
            chunk_seconds=chunk_seconds,
            overlap_ratio=0.5,
            sample_rate=rate,
            device=device,
            stem_names=stem_names,
            progress_callback=progress_callback,
        )

    return estimator
