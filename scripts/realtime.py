"""Run realtime ASL inference with a trained KNN model."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asl.landmarks import extract_landmarks, landmarks_to_row

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def ensure_hand_model(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {path}...")
    urllib.request.urlretrieve(HAND_MODEL_URL, path)


def draw_hand_landmarks(frame, hand_landmarks) -> None:
    height, width = frame.shape[:2]
    for start, end in HAND_CONNECTIONS:
        start_point = hand_landmarks[start]
        end_point = hand_landmarks[end]
        x1, y1 = int(start_point.x * width), int(start_point.y * height)
        x2, y2 = int(end_point.x * width), int(end_point.y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for landmark in hand_landmarks:
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime ASL inference from webcam.")
    parser.add_argument("--model", default="models/asl_knn.npz", help="Trained model path.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence threshold.")
    parser.add_argument("--smooth-window", type=int, default=7, help="Prediction smoothing window.")
    parser.add_argument("--spell", action="store_true", help="Accumulate letters into a word.")
    parser.add_argument("--stable-frames", type=int, default=5, help="Frames required to emit a letter.")
    parser.add_argument(
        "--hand-model",
        default="models/hand_landmarker.task",
        help="Path to the MediaPipe hand landmark model.",
    )
    return parser.parse_args()


def smooth_prediction(history: deque[tuple[str, float]]) -> tuple[str, float] | None:
    if not history:
        return None
    counter = Counter(label for label, _ in history)
    top_label, _ = counter.most_common(1)[0]
    confidences = [conf for label, conf in history if label == top_label]
    avg_conf = float(sum(confidences) / max(len(confidences), 1))
    return top_label, avg_conf


def knn_predict(
    x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k: int
) -> tuple[str, float]:
    dists = np.linalg.norm(X_train - x, axis=1)
    k = max(1, min(k, len(dists)))
    idx = np.argpartition(dists, k - 1)[:k]

    weights = 1.0 / (dists[idx] + 1e-6)
    scores: dict[str, float] = {}
    for label, weight in zip(y_train[idx], weights):
        scores[label] = scores.get(label, 0.0) + float(weight)

    top_label = max(scores.items(), key=lambda item: item[1])[0]
    conf = scores[top_label] / max(sum(scores.values()), 1e-6)
    return top_label, float(conf)


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1
    hand_model_path = Path(args.hand_model)
    ensure_hand_model(hand_model_path)

    model = np.load(model_path, allow_pickle=False)
    X_train = model["X"]
    y_train = model["y"]
    mean = model["mean"]
    scale = model["scale"]
    k = int(model["k"][0]) if "k" in model else 5

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera.")
        return 1

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    history: deque[tuple[str, float]] = deque(maxlen=max(args.smooth_window, 1))
    text = ""
    last_label = None
    stable_count = 0
    last_emitted = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            draw_hand_landmarks(frame, hand_landmarks)

            row = landmarks_to_row(extract_landmarks(hand_landmarks))
            if row:
                x = (np.array(row, dtype=np.float32) - mean) / scale
                label, conf = knn_predict(x, X_train, y_train, k)
                history.append((label, conf))
        else:
            history.clear()

        smoothed = smooth_prediction(history)
        if smoothed:
            label, conf = smoothed
            status = f"{label} ({conf:.2f})"
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 0), 2)

            if label == last_label:
                stable_count += 1
            else:
                stable_count = 1
                last_label = label

            if args.spell and conf >= args.min_confidence:
                if stable_count >= args.stable_frames and label != last_emitted:
                    if label == "SPACE":
                        text += " "
                        last_emitted = label
                    elif label == "DEL":
                        text = text[:-1]
                        last_emitted = label
                    elif len(label) == 1 and label.isalnum():
                        text += label
                        last_emitted = label
        else:
            cv2.putText(frame, "No prediction", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)

        if args.spell:
            cv2.putText(frame, f"Text: {text}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
            cv2.putText(frame, "Use SPACE or DEL labels to edit", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, "q=quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("ASL Realtime", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
