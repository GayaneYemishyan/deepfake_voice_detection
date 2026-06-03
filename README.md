# 🛡️ VoiceShield — Deepfake Voice Detection

A machine learning system that classifies audio as **REAL** or **FAKE** using a 227-dimensional acoustic feature pipeline and a PyTorch MLP, served via a Flask web app.

---

## Repository Structure

```
deepfake_voice_detection/
├── deepfake-voice-detection.ipynb   # Training, feature engineering & model comparison
└── UI/
    ├── app.py                       # Flask backend — feature extraction + MLP inference
    ├── requirements.txt
    ├── models/
    │   ├── mlp_best.pt              # Trained MLP weights (PyTorch)
    │   ├── scaler.pkl               # StandardScaler fitted on training data
    │   ├── clip_bounds.pkl          # Per-feature [q01, q99] clipping bounds
    │   └── optimal_threshold.pkl   # Decision threshold (tuned on validation set)
    └── static/
        ├── index.html               # Web UI
        └── logo.png
```

---

## How It Works

### 1. Feature Extraction

Every audio file is resampled to **16 kHz mono**, normalised, trimmed of silence, and then a **227-dimensional feature vector** is extracted using [librosa](https://librosa.org/):

| Group | Features | Dims |
|---|---|---|
| Spectral shape | Centroid, bandwidth, roll-off, flatness (mean + std) | 8 |
| Spectral contrast | 7 sub-bands (mean + std) | 14 |
| Chroma STFT | 12 pitch classes (mean + std) | 24 |
| Tonnetz | 6 tonal centroid features (mean + std) | 12 |
| GFCCs | 20 gammatone-filtered cepstral coefficients (mean + std) | 40 |
| Delta MFCCs | 1st-order MFCC deltas, 20 coefficients (mean + std) | 40 |
| Delta-delta MFCCs | 2nd-order MFCC deltas, 20 coefficients (mean + std) | 40 |
| Prosody (F0) | Mean, std, range, voiced fraction via pYIN | 4 |
| Energy & timing | ZCR, RMS (mean + std), HNR approximation | 5 |
| MFCCs | 20 mean coefficients | 20 |
| LFCCs | 20 linear-filter cepstral coefficients | 20 |
| **Total** | | **227** |

### 2. Preprocessing

Features are clipped to per-feature `[q01, q99]` bounds (stored in `clip_bounds.pkl`) and then standardised with a `StandardScaler` (`scaler.pkl`) before being fed to the model.

### 3. Model — DeepfakeMLP

A fully-connected network implemented in PyTorch:

```
Input (227)
  → Linear(227→512) → BatchNorm1d → GELU → Dropout(0.3)
  → Linear(512→256) → BatchNorm1d → GELU → Dropout(0.3)
  → Linear(256→128) → BatchNorm1d → GELU → Dropout(0.3)
  → Linear(128→64)  → BatchNorm1d → GELU → Dropout(0.3)
  → Linear(64→1)    → Sigmoid
```

The decision threshold is learned on the validation set and stored in `optimal_threshold.pkl` rather than hardcoded to 0.5.

### 4. Windowed Timeline Analysis

Alongside a clip-level verdict, the pipeline splits the audio into **2-second windows with a 0.5-second hop** and runs a single batched forward pass across all windows (3–8× faster than serial inference). The response includes a per-window `timeline` array so the UI can highlight which segments of the audio are suspicious.

---

## API

### `POST /predict`

Upload an audio file and receive a prediction.

**Accepted formats:** `wav`, `mp3`, `flac`, `ogg`, `m4a`, `aac`, `opus`, `webm`

**Request:**
```bash
curl -X POST http://localhost:7860/predict \
     -F "file=@sample.wav"
```

**Response:**
```json
{
  "label": "FAKE",
  "is_fake": true,
  "raw_probability": 0.913,
  "confidence_pct": 91.3,
  "threshold": 0.4821,
  "duration_sec": 6.4,
  "window_sec": 2.0,
  "hop_sec": 0.5,
  "overall_prob": 0.8942,
  "timeline": [
    { "time_start": 0.0, "time_end": 2.0, "time_mid": 1.0, "prob": 0.871, "is_fake": true },
    { "time_start": 0.5, "time_end": 2.5, "time_mid": 1.5, "prob": 0.904, "is_fake": true }
  ]
}
```

### `GET /health`

```json
{ "status": "ok", "features": 227, "threshold": 0.4821 }
```

---

## Local Setup

```bash
# Clone
git clone https://github.com/GayaneYemishyan/deepfake_voice_detection.git
cd deepfake_voice_detection/UI

# Install dependencies
pip install -r requirements.txt

# Run the server (port 7860)
python app.py
```

Then open `http://localhost:7860` in your browser.

---

## Training

The full training pipeline — data loading, feature extraction, model comparison (Random Forest, LightGBM, XGBoost, MLP), hyperparameter tuning, and threshold optimisation — is documented in:

```
deepfake-voice-detection.ipynb
```

The notebook produces all four artefacts saved under `UI/models/`.

---

## Dependencies

Key packages (see `UI/requirements.txt` for pinned versions):

- `torch` — MLP definition and inference
- `librosa` — audio loading and all feature extraction
- `joblib` — loading scaler, clip bounds, threshold
- `flask` + `flask-cors` — REST API and static file serving
- `numpy` — numerical operations

---

## Acknowledgements

**Advisor:** Arsen Davtyan
