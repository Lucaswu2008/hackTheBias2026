"""Landmark extraction and normalization helpers."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


Landmark = Tuple[float, float, float]


def extract_landmarks(hand_landmarks: object) -> List[Landmark]:
    """Extract (x, y, z) tuples from MediaPipe hand landmarks."""
    points = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    return [(lm.x, lm.y, lm.z) for lm in points]


def normalize_landmarks(landmarks: Sequence[Landmark]) -> Optional[np.ndarray]:
    """Normalize landmarks relative to wrist and scale to unit max distance."""
    if len(landmarks) != 21:
        return None

    arr = np.array(landmarks, dtype=np.float32)
    wrist = arr[0]
    arr = arr - wrist

    scale = float(np.max(np.linalg.norm(arr, axis=1)))
    if scale < 1e-6:
        return None

    arr = arr / scale
    return arr.reshape(-1)


def landmarks_to_row(landmarks: Sequence[Landmark]) -> Optional[List[float]]:
    """Convert landmarks into a flat feature row."""
    normalized = normalize_landmarks(landmarks)
    if normalized is None:
        return None
    return normalized.astype(float).tolist()
