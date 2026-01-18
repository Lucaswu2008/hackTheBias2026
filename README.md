# hackTheBias2026

Realtime ASL detection pipeline using MediaPipe hand landmarks and a lightweight
classifier. It supports letters and words as long as you collect training data
for those labels.

## Setup

Use Python 3.12 (MediaPipe does not yet support Python 3.14).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Collect training data

Capture samples for each sign (letters or words). Repeat per label. Collection
starts OFF by default; press `c` to toggle on.

```powershell
python scripts/collect_landmarks.py --label A --samples 200
python scripts/collect_landmarks.py --label HELLO --samples 200
```

Controls:
- `c` toggle continuous capture
- `s` save single sample
- `r` clear samples from the current session
- `x` clear all samples in the CSV
- `q` quit

Tips: keep the hand centered, vary distance/rotation, and collect in different
lighting to improve robustness. Default capture interval is 0.25s (override
with `--interval`).

## 2) Train the model

```powershell
python scripts/train.py --data data/landmarks.csv
```

Outputs:
- `models/asl_knn.npz`
- `models/labels.json`

## 3) Run realtime inference

```powershell
python scripts/realtime.py --model models/asl_knn.npz
```

If you want to spell words from letters, include labels like `A`-`Z` and
optional `SPACE`/`DEL`, then run:

```powershell
python scripts/realtime.py --spell
```

## Notes
- To embed the detector in the website lesson pages, start the backend:
  ```powershell
  .\.venv\Scripts\python backend\app.py
  ```
  Then open `web/index.html`. The lesson practice pages load
  `http://localhost:5000/detect?letter=A` (and other letters) inside the iframe.
- The first run downloads `models/hand_landmarker.task` automatically (override with `--hand-model`).
- The classifier predicts whatever labels you trained on (letters or full words).
- For best accuracy, record at least a few hundred samples per label.
