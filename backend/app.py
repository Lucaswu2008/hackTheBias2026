from __future__ import annotations

import html
import time
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

import sys

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

app = Flask(__name__)
lock = Lock()
GLOBAL_STATUS_KEY = "__global__"
status_state: dict[str, dict[str, float | bool | str | None]] = {}


def ensure_hand_model(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)


def load_model(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    data = np.load(path, allow_pickle=False)
    return {
        "X": data["X"],
        "y": data["y"],
        "mean": data["mean"],
        "scale": data["scale"],
        "k": data["k"][0] if "k" in data else 5,
    }


def knn_predict(x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k: int) -> tuple[str, float]:
    dists = np.linalg.norm(X_train - x, axis=1)
    k = max(1, min(int(k), len(dists)))
    idx = np.argpartition(dists, k - 1)[:k]

    weights = 1.0 / (dists[idx] + 1e-6)
    scores: dict[str, float] = {}
    for label, weight in zip(y_train[idx], weights):
        scores[label] = scores.get(label, 0.0) + float(weight)

    top_label = max(scores.items(), key=lambda item: item[1])[0]
    conf = scores[top_label] / max(sum(scores.values()), 1e-6)
    return top_label, float(conf)


def init_landmarker(hand_model_path: Path) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


hand_model_path = ROOT / "models" / "hand_landmarker.task"
ensure_hand_model(hand_model_path)
landmarker = init_landmarker(hand_model_path)
model = load_model(ROOT / "models" / "asl_knn.npz")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def generate_frames(target_letter: str | None):
    target_upper = target_letter.upper() if target_letter else None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.time()
        with lock:
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            results = landmarker.detect_for_video(mp_image, int(now * 1000))

        label = None
        conf = None
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            draw_hand_landmarks(frame, hand_landmarks)
            row = landmarks_to_row(extract_landmarks(hand_landmarks))
            if row:
                x = (np.array(row, dtype=np.float32) - model["mean"]) / model["scale"]
                label, conf = knn_predict(x, model["X"], model["y"], model["k"])

        accuracy_pct = 0.0
        if label and conf is not None:
            if target_upper and label != target_upper:
                accuracy_pct = 0.0
            else:
                accuracy_pct = max(0.0, min(conf * 100.0, 100.0))

        target_key = target_upper or "_"
        with lock:
            state = status_state.setdefault(
                target_key,
                {"stable_start": None, "target_met": False, "accuracy": 0.0, "stable_seconds": 0.0},
            )

            if target_upper and label == target_upper:
                if state["stable_start"] is None:
                    state["stable_start"] = now
                stable_seconds = float(now - (state["stable_start"] or now))
                target_met = stable_seconds >= 3.0
            else:
                state["stable_start"] = None
                stable_seconds = 0.0
                target_met = False

            state["accuracy"] = float(accuracy_pct)
            state["target_met"] = bool(target_met)
            state["stable_seconds"] = float(stable_seconds)

            global_state = status_state.setdefault(
                GLOBAL_STATUS_KEY,
                {"label": None, "label_stable_start": None, "label_stable_seconds": 0.0},
            )
            if label:
                if global_state.get("label") != label:
                    global_state["label"] = label
                    global_state["label_stable_start"] = now
                    global_state["label_stable_seconds"] = 0.0
                else:
                    if global_state.get("label_stable_start") is None:
                        global_state["label_stable_start"] = now
                    global_state["label_stable_seconds"] = float(
                        now - (global_state.get("label_stable_start") or now)
                    )
            else:
                global_state["label"] = None
                global_state["label_stable_start"] = None
                global_state["label_stable_seconds"] = 0.0

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/detect")
def detect():
    letter = request.args.get("letter", "")
    safe_letter = html.escape(letter)
    feed_url = "/video_feed?letter=" + urllib.parse.quote(letter)
    return (
        "<!doctype html>"
        "<html lang=\"en\">"
        "<head>"
        "<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Detector</title>"
        "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">"
        "<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>"
        "<link href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap\" rel=\"stylesheet\">"
        "<style>"
        "html,body{height:100%;overflow:hidden;}"
        "body{margin:0;background:#ffffff;font-family:'IBM Plex Sans',sans-serif;}"
        ".wrap{display:grid;place-items:center;height:100%;padding:8px;}"
        ".card{width:min(95vw,820px);height:auto;display:flex;flex-direction:column;}"
        ".frame{border:1px solid #cbcbcb;border-radius:16px;overflow:hidden;background:#f3f3f3;"
        "width:100%;aspect-ratio:16/9;margin:0 12px 8px;}"
        "img{width:100%;height:100%;object-fit:contain;display:block;}"
        "</style>"
        "</head>"
        "<body>"
        "<div class=\"wrap\">"
        "<div class=\"card\">"
        f"<div class=\"frame\"><img src=\"{feed_url}\" alt=\"Detector feed\"></div>"
        "</div>"
        "</div>"
        "<script>"
        "const params=new URLSearchParams(window.location.search);"
        "const letter=params.get('letter')||'';"
        "const poll=()=>{"
        "fetch(`/status?letter=${encodeURIComponent(letter)}`)"
        ".then(res=>res.json())"
        ".then(data=>{"
        "const accuracy=Math.round(data.accuracy||0);"
        "window.parent.postMessage({"
        "type:'detector-status',"
        "letter:letter,"
        "accuracy:accuracy,"
        "targetMet:!!data.target_met"
        "},'*');"
        "})"
        ".catch(()=>{});"
        "};"
        "poll();"
        "setInterval(poll,500);"
        "</script>"
        "</body>"
        "</html>"
    )


@app.route("/video_feed")
def video_feed():
    letter = request.args.get("letter")
    return Response(
        generate_frames(letter),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    letter = (request.args.get("letter") or "").upper()
    key = letter or "_"
    with lock:
        state = status_state.get(key, {})
        global_state = status_state.get(GLOBAL_STATUS_KEY, {})
        payload = {
            "accuracy": state.get("accuracy", 0.0),
            "target_met": state.get("target_met", False),
            "stable_seconds": state.get("stable_seconds", 0.0),
            "label": global_state.get("label"),
            "label_stable_seconds": global_state.get("label_stable_seconds", 0.0),
        }
    response = jsonify(payload)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
