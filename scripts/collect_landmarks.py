"""Collect ASL hand landmarks for training."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

import cv2
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


def _build_header() -> list[str]:
    return ["label"] + [f"f{i}" for i in range(63)]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_header_if_needed(path: Path, writer: csv.writer) -> None:
    if not path.exists() or path.stat().st_size == 0:
        writer.writerow(_build_header())


def _count_label_samples(path: Path, label: str) -> int:
    if not path.exists():
        return 0

    lines = path.read_text(encoding="ascii").splitlines()
    if not lines:
        return 0

    start_index = 1 if lines[0].lower().startswith("label,") else 0
    prefix = f"{label},"
    return sum(1 for line in lines[start_index:] if line.startswith(prefix))


def _count_all_labels(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}

    lines = path.read_text(encoding="ascii").splitlines()
    if not lines:
        return {}

    start_index = 1 if lines[0].lower().startswith("label,") else 0
    counts: dict[str, int] = {}
    for line in lines[start_index:]:
        if not line:
            continue
        label = line.split(",", 1)[0]
        counts[label] = counts.get(label, 0) + 1
    return counts


def _format_label_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["Totals: none"]

    items = [f"{label}={counts[label]}" for label in sorted(counts.keys())]
    lines = ["Totals:"]
    line = []
    for item in items:
        line.append(item)
        if len(line) == 6:
            lines.append("  ".join(line))
            line = []
    if line:
        lines.append("  ".join(line))
    return lines


def _remove_last_samples(path: Path, count: int) -> int:
    if count <= 0 or not path.exists():
        return 0

    lines = path.read_text(encoding="ascii").splitlines()
    if not lines:
        return 0

    if lines[0].lower().startswith("label,"):
        header = lines[0]
        data_lines = lines[1:]
    else:
        header = ",".join(_build_header())
        data_lines = lines

    remove_count = min(count, len(data_lines))
    remaining = data_lines[:-remove_count] if remove_count else data_lines
    path.write_text("\n".join([header] + remaining) + "\n", encoding="ascii")
    return remove_count


def _clear_all_samples(path: Path) -> None:
    header = ",".join(_build_header())
    path.write_text(header + "\n", encoding="ascii")


def _open_writer(path: Path) -> tuple[csv.writer, object]:
    csv_file = path.open("a", newline="", encoding="ascii")
    writer = csv.writer(csv_file)
    _write_header_if_needed(path, writer)
    csv_file.flush()
    return writer, csv_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ASL hand landmarks for training.")
    parser.add_argument("--label", required=True, help="Label for the captured sign (letter or word).")
    parser.add_argument("--output", default="data/landmarks.csv", help="CSV output path.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--samples", type=int, default=200, help="Target samples to capture while collecting.")
    parser.add_argument("--interval", type=float, default=0.25, help="Minimum seconds between samples.")
    parser.add_argument("--autostart", action="store_true", help="Start collecting immediately.")
    parser.add_argument(
        "--hand-model",
        default="models/hand_landmarker.task",
        help="Path to the MediaPipe hand landmark model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_path = Path(args.output)
    _ensure_parent(output_path)
    hand_model_path = Path(args.hand_model)
    ensure_hand_model(hand_model_path)

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

    collecting = args.autostart
    target_samples = max(args.samples, 0)
    samples = 0
    last_capture = 0.0

    writer, csv_file = _open_writer(output_path)
    label_counts = _count_all_labels(output_path)
    total_samples = label_counts.get(args.label, 0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            row = None
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                draw_hand_landmarks(frame, hand_landmarks)
                row = landmarks_to_row(extract_landmarks(hand_landmarks))

            if row and collecting:
                now = time.time()
                if now - last_capture >= args.interval:
                    writer.writerow([args.label] + row)
                    csv_file.flush()
                    samples += 1
                    total_samples += 1
                    label_counts[args.label] = total_samples
                    last_capture = now
                    if target_samples and samples >= target_samples:
                        collecting = False

            status = "ON" if collecting else "OFF"
            message = (
                f"Label: {args.label} | Session: {samples} | Total: {total_samples} | Collect: {status}"
            )
            cv2.putText(frame, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 0), 2)
            cv2.putText(
                frame,
                "c=toggle s=save r=clear-session x=clear-all q=quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
            )

            if not results.hand_landmarks:
                cv2.putText(frame, "No hand detected", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

            y_offset = 105
            for line in _format_label_counts(label_counts):
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                y_offset += 18

            cv2.imshow("ASL Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                collecting = not collecting
            if key == ord("s") and row:
                writer.writerow([args.label] + row)
                csv_file.flush()
                samples += 1
                total_samples += 1
                label_counts[args.label] = total_samples
            if key == ord("r"):
                if samples > 0:
                    csv_file.flush()
                    csv_file.close()
                    removed = _remove_last_samples(output_path, samples)
                    samples = max(samples - removed, 0)
                    total_samples = max(total_samples - removed, 0)
                    if total_samples > 0:
                        label_counts[args.label] = total_samples
                    else:
                        label_counts.pop(args.label, None)
                    writer, csv_file = _open_writer(output_path)
            if key == ord("x"):
                csv_file.flush()
                csv_file.close()
                _clear_all_samples(output_path)
                samples = 0
                total_samples = 0
                label_counts = {}
                writer, csv_file = _open_writer(output_path)

    finally:
        csv_file.close()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
