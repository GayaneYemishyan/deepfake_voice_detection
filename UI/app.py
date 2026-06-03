import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import joblib
import librosa
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

FEATURE_COLS = [
    'spectral_centroid_mean','spectral_centroid_std',
    'spectral_bandwidth_mean','spectral_bandwidth_std',
    'spectral_rolloff_mean','spectral_rolloff_std',
    'spectral_flatness_mean','spectral_flatness_std',
    'spectral_contrast_1_mean','spectral_contrast_1_std',
    'spectral_contrast_2_mean','spectral_contrast_2_std',
    'spectral_contrast_3_mean','spectral_contrast_3_std',
    'spectral_contrast_4_mean','spectral_contrast_4_std',
    'spectral_contrast_5_mean','spectral_contrast_5_std',
    'spectral_contrast_6_mean','spectral_contrast_6_std',
    'spectral_contrast_7_mean','spectral_contrast_7_std',
    'chroma_1_mean','chroma_1_std','chroma_2_mean','chroma_2_std',
    'chroma_3_mean','chroma_3_std','chroma_4_mean','chroma_4_std',
    'chroma_5_mean','chroma_5_std','chroma_6_mean','chroma_6_std',
    'chroma_7_mean','chroma_7_std','chroma_8_mean','chroma_8_std',
    'chroma_9_mean','chroma_9_std','chroma_10_mean','chroma_10_std',
    'chroma_11_mean','chroma_11_std','chroma_12_mean','chroma_12_std',
    'tonnetz_1_mean','tonnetz_1_std','tonnetz_2_mean','tonnetz_2_std',
    'tonnetz_3_mean','tonnetz_3_std','tonnetz_4_mean','tonnetz_4_std',
    'tonnetz_5_mean','tonnetz_5_std','tonnetz_6_mean','tonnetz_6_std',
    'gfcc_1_mean','gfcc_1_std','gfcc_2_mean','gfcc_2_std',
    'gfcc_3_mean','gfcc_3_std','gfcc_4_mean','gfcc_4_std',
    'gfcc_5_mean','gfcc_5_std','gfcc_6_mean','gfcc_6_std',
    'gfcc_7_mean','gfcc_7_std','gfcc_8_mean','gfcc_8_std',
    'gfcc_9_mean','gfcc_9_std','gfcc_10_mean','gfcc_10_std',
    'gfcc_11_mean','gfcc_11_std','gfcc_12_mean','gfcc_12_std',
    'gfcc_13_mean','gfcc_13_std','gfcc_14_mean','gfcc_14_std',
    'gfcc_15_mean','gfcc_15_std','gfcc_16_mean','gfcc_16_std',
    'gfcc_17_mean','gfcc_17_std','gfcc_18_mean','gfcc_18_std',
    'gfcc_19_mean','gfcc_19_std','gfcc_20_mean','gfcc_20_std',
    'delta_mfcc_1_mean','delta_mfcc_1_std','delta2_mfcc_1_mean','delta2_mfcc_1_std',
    'delta_mfcc_2_mean','delta_mfcc_2_std','delta2_mfcc_2_mean','delta2_mfcc_2_std',
    'delta_mfcc_3_mean','delta_mfcc_3_std','delta2_mfcc_3_mean','delta2_mfcc_3_std',
    'delta_mfcc_4_mean','delta_mfcc_4_std','delta2_mfcc_4_mean','delta2_mfcc_4_std',
    'delta_mfcc_5_mean','delta_mfcc_5_std','delta2_mfcc_5_mean','delta2_mfcc_5_std',
    'delta_mfcc_6_mean','delta_mfcc_6_std','delta2_mfcc_6_mean','delta2_mfcc_6_std',
    'delta_mfcc_7_mean','delta_mfcc_7_std','delta2_mfcc_7_mean','delta2_mfcc_7_std',
    'delta_mfcc_8_mean','delta_mfcc_8_std','delta2_mfcc_8_mean','delta2_mfcc_8_std',
    'delta_mfcc_9_mean','delta_mfcc_9_std','delta2_mfcc_9_mean','delta2_mfcc_9_std',
    'delta_mfcc_10_mean','delta_mfcc_10_std','delta2_mfcc_10_mean','delta2_mfcc_10_std',
    'delta_mfcc_11_mean','delta_mfcc_11_std','delta2_mfcc_11_mean','delta2_mfcc_11_std',
    'delta_mfcc_12_mean','delta_mfcc_12_std','delta2_mfcc_12_mean','delta2_mfcc_12_std',
    'delta_mfcc_13_mean','delta_mfcc_13_std','delta2_mfcc_13_mean','delta2_mfcc_13_std',
    'delta_mfcc_14_mean','delta_mfcc_14_std','delta2_mfcc_14_mean','delta2_mfcc_14_std',
    'delta_mfcc_15_mean','delta_mfcc_15_std','delta2_mfcc_15_mean','delta2_mfcc_15_std',
    'delta_mfcc_16_mean','delta_mfcc_16_std','delta2_mfcc_16_mean','delta2_mfcc_16_std',
    'delta_mfcc_17_mean','delta_mfcc_17_std','delta2_mfcc_17_mean','delta2_mfcc_17_std',
    'delta_mfcc_18_mean','delta_mfcc_18_std','delta2_mfcc_18_mean','delta2_mfcc_18_std',
    'delta_mfcc_19_mean','delta_mfcc_19_std','delta2_mfcc_19_mean','delta2_mfcc_19_std',
    'delta_mfcc_20_mean','delta_mfcc_20_std','delta2_mfcc_20_mean','delta2_mfcc_20_std',
    'f0_mean','f0_std','f0_range','voiced_fraction',
    'zcr_mean','zcr_std','rms_mean','rms_std','hnr_approx',
    'mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5',
    'mfcc_6','mfcc_7','mfcc_8','mfcc_9','mfcc_10',
    'mfcc_11','mfcc_12','mfcc_13','mfcc_14','mfcc_15',
    'mfcc_16','mfcc_17','mfcc_18','mfcc_19','mfcc_20',
    'lfcc_1','lfcc_2','lfcc_3','lfcc_4','lfcc_5',
    'lfcc_6','lfcc_7','lfcc_8','lfcc_9','lfcc_10',
    'lfcc_11','lfcc_12','lfcc_13','lfcc_14','lfcc_15',
    'lfcc_16','lfcc_17','lfcc_18','lfcc_19','lfcc_20',
]

N_FEATURES        = len(FEATURE_COLS)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'opus', 'webm'}
WINDOW_SEC        = 2.0
HOP_SEC           = 0.5


class DeepfakeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(512, 256, 128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

DEVICE    = torch.device('cpu')
scaler    = joblib.load('models/scaler.pkl')
q01, q99  = joblib.load('models/clip_bounds.pkl')
threshold = joblib.load('models/optimal_threshold.pkl')

mlp = DeepfakeMLP(in_dim=N_FEATURES).to(DEVICE)
mlp.load_state_dict(torch.load('models/mlp_best.pt', map_location=DEVICE))
mlp.eval()

print(f"[startup] MLP loaded | threshold={threshold:.4f} | features={N_FEATURES}")

def load_audio(path: str):
    y_native, sr_orig = librosa.load(path, sr=None, mono=True)
    return librosa.resample(y_native, orig_sr=sr_orig, target_sr=16000), 16000

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    if np.percentile(np.abs(y), 5) < 0.001:
        y = y + np.random.normal(0, 0.002, y.shape)
        y = librosa.effects.preemphasis(y, coef=0.85)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.9
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))

    features = {}

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = sc.mean(); features['spectral_centroid_std'] = sc.std()
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = sb.mean(); features['spectral_bandwidth_std'] = sb.std()
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = sr_.mean(); features['spectral_rolloff_std'] = sr_.std()
    sf = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = sf.mean(); features['spectral_flatness_std'] = sf.std()

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(7):
        features[f'spectral_contrast_{i+1}_mean'] = contrast[i].mean()
        features[f'spectral_contrast_{i+1}_std']  = contrast[i].std()

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f'chroma_{i+1}_mean'] = chroma[i].mean()
        features[f'chroma_{i+1}_std']  = chroma[i].std()

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(6):
        features[f'tonnetz_{i+1}_mean'] = tonnetz[i].mean()
        features[f'tonnetz_{i+1}_std']  = tonnetz[i].std()

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    gfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=20)
    for i in range(20):
        features[f'gfcc_{i+1}_mean'] = gfcc[i].mean()
        features[f'gfcc_{i+1}_std']  = gfcc[i].std()

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i in range(20):
        features[f'mfcc_{i+1}']             = mfcc[i].mean()
        features[f'delta_mfcc_{i+1}_mean']  = delta_mfcc[i].mean()
        features[f'delta_mfcc_{i+1}_std']   = delta_mfcc[i].std()
        features[f'delta2_mfcc_{i+1}_mean'] = delta2_mfcc[i].mean()
        features[f'delta2_mfcc_{i+1}_std']  = delta2_mfcc[i].std()

    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([0.0])
    f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
    if len(f0_voiced) == 0:
        f0_voiced = np.array([0.0])
    features['f0_mean']         = float(f0_voiced.mean())
    features['f0_std']          = float(f0_voiced.std())
    features['f0_range']        = float(f0_voiced.max() - f0_voiced.min())
    features['voiced_fraction'] = float(voiced_flag.sum() / len(voiced_flag)) if voiced_flag is not None else 0.0

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = zcr.mean(); features['zcr_std'] = zcr.std()
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = rms.mean(); features['rms_std'] = rms.std()

    harmonic, percussive = librosa.effects.hpss(y)
    features['hnr_approx'] = float(10 * np.log10(
        (np.mean(harmonic**2) + 1e-10) / (np.mean(percussive**2) + 1e-10)
    ))

    stft = np.abs(librosa.stft(y))
    lin_filters = librosa.filters.mel(sr=sr, n_fft=stft.shape[0]*2-2, n_mels=20, htk=True)
    lfcc = librosa.feature.mfcc(S=librosa.power_to_db(np.dot(lin_filters, stft) + 1e-10), n_mfcc=20)
    for i in range(20):
        features[f'lfcc_{i+1}'] = lfcc[i].mean()

    return np.array([features[f] for f in FEATURE_COLS], dtype=np.float32)

def feats_to_prob(feats: np.ndarray) -> float:
    fc = np.clip(feats, q01, q99)
    fs = scaler.transform(fc.reshape(1, -1))
    with torch.no_grad():
        return torch.sigmoid(mlp(torch.from_numpy(fs).float().to(DEVICE))).item()

def _batch_feats_to_probs(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Run a single batched forward pass for an (N, F) feature matrix.
    Returns an (N,) numpy array of probabilities.
    Replaces N serial feats_to_prob calls with one batched inference call,
    typically 3-8x faster for the window counts used in run_pipeline.
    """
    mat_clipped = np.clip(feature_matrix, q01, q99)
    mat_scaled  = scaler.transform(mat_clipped).astype(np.float32)
    with torch.no_grad():
        logits = mlp(torch.from_numpy(mat_scaled).to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
    return probs  # shape: (N,)


def run_pipeline(audio_path: str) -> dict:
    """Full predict + timeline on a local audio file. Returns combined dict."""
    y, sr    = load_audio(audio_path)
    duration = len(y) / sr

    # --- overall clip-level prediction (single sample, unchanged) ---
    feats   = extract_features(y, sr)
    prob    = feats_to_prob(feats)
    is_fake = prob >= threshold

    # --- window setup ---
    TARGET_WINDOWS = 6
    if duration >= WINDOW_SEC:
        win_sec = WINDOW_SEC
        hop_sec = HOP_SEC
    else:
        win_sec = max(duration / TARGET_WINDOWS, 0.1)
        hop_sec = win_sec / 2

    win_samples = int(win_sec * sr)
    hop_samples = max(int(hop_sec * sr), 1)

    # --- collect all window feature vectors first, then batch-infer ---
    window_feats, window_metas = [], []
    start = 0
    while start + win_samples <= len(y):
        chunk = y[start: start + win_samples]
        t_s   = start / sr
        try:
            window_feats.append(extract_features(chunk, sr))
        except Exception:
            window_feats.append(np.zeros(N_FEATURES, dtype=np.float32))
        window_metas.append({
            'time_start': round(t_s, 2),
            'time_end':   round(t_s + win_sec, 2),
            'time_mid':   round(t_s + win_sec / 2, 2),
        })
        start += hop_samples

    # single batched forward pass replaces N serial feats_to_prob calls
    timeline = []
    if window_feats:
        feature_matrix = np.stack(window_feats)           # (N, F)
        batch_probs    = _batch_feats_to_probs(feature_matrix)  # (N,)
        for meta, p in zip(window_metas, batch_probs):
            p = float(p)
            timeline.append({
                **meta,
                'prob':    round(p, 4),
                'is_fake': bool(p >= threshold),
            })

    if not timeline:
        timeline.append({
            'time_start': 0.0, 'time_end': round(duration, 2),
            'time_mid': round(duration / 2, 2),
            'prob': round(prob, 4), 'is_fake': bool(is_fake),
        })

    return {
        'is_fake':         bool(is_fake),
        'raw_probability': round(prob, 6),
        'threshold':       round(float(threshold), 6),
        'confidence_pct':  round((prob if is_fake else 1 - prob) * 100, 2),
        'label':           'FAKE' if is_fake else 'REAL',
        'duration_sec':    round(duration, 2),
        'window_sec':      round(win_sec, 3),
        'hop_sec':         round(hop_sec, 3),
        'timeline':        timeline,
        'overall_prob':    round(float(np.mean([w['prob'] for w in timeline])), 4),
    }

app = Flask(__name__, static_folder='static')
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_upload(f) -> str:
    suffix   = '.' + secure_filename(f.filename).rsplit('.', 1)[-1]
    tmp      = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    f.save(tmp_path)
    tmp.close()
    return tmp_path

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if f.filename == '' or not allowed_file(f.filename):
        return jsonify({'error': 'Invalid file'}), 400

    tmp_path = save_upload(f)
    try:
        return jsonify(run_pipeline(tmp_path))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'features': N_FEATURES, 'threshold': float(threshold)})

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=7860, debug=False)