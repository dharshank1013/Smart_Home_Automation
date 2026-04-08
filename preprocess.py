"""
preprocess.py
Audio and text pre-processing utilities used by the pipeline.
"""

import re
import numpy as np


# ── Text cleaning ──────────────────────────────────────────────────────────────

FILLER_WORDS = {"um", "uh", "er", "ah", "like", "you know", "so", "well"}


def clean_transcript(text: str) -> str:
    """Remove STT artefacts and normalise text."""
    text = text.strip()
    # Remove repeated words (common STT glitch)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.I)
    # Remove leading filler words
    for fw in FILLER_WORDS:
        text = re.sub(rf'^\s*{fw}\s*,?\s*', '', text, flags=re.I)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalise_command(text: str) -> str:
    """Lower-case and strip punctuation for intent matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Audio helpers ──────────────────────────────────────────────────────────────

def resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Simple linear resample (use librosa for production)."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_samples),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def normalise_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalise audio to [-1, 1]."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio


def remove_silence(audio: np.ndarray, sr: int = 16000,
                   threshold: float = 0.01, min_silence_ms: int = 300) -> np.ndarray:
    """Strip leading/trailing silence."""
    min_samples = int(sr * min_silence_ms / 1000)
    mask = np.abs(audio) > threshold
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return audio
    start = max(0, indices[0] - min_samples)
    end   = min(len(audio), indices[-1] + min_samples)
    return audio[start:end]


def chunk_audio(audio: np.ndarray, sr: int = 16000,
                chunk_sec: float = 30.0) -> list:
    """Split long audio into Whisper-friendly chunks (<= 30 s)."""
    chunk_size = int(sr * chunk_sec)
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
