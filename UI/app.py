import json
import joblib
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
import gradio as gr
from spafe.features.lfcc import lfcc as compute_lfcc

warnings.filterwarnings('ignore')

# ── Model definition (must match training exactly) ─────────────────────────
# ── Model definition — reads architecture FROM checkpoint, no hardcoding ──
class DeepfakeDetectorMLP(nn.Module):
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of (in, out) tuples for Linear layers,
                     inferred automatically from checkpoint.
        """
        super().__init__()
        layers = []
        dropouts = [0.3, 0.3, 0.2, 0.1]  # must match training order
        for i, (in_f, out_f) in enumerate(layer_sizes[:-1]):
            layers += [
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.GELU(),
                nn.Dropout(dropouts[i] if i < len(dropouts) else 0.1),
            ]
        # Final layer — no BN/activation/dropout
        layers.append(nn.Linear(layer_sizes[-1][0], layer_sizes[-1][1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_mlp_from_checkpoint(path, device):
    """Infer architecture from saved weights and build matching model."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd   = ckpt['model_state']

    # Find all Linear layers by scanning state_dict keys
    linear_keys = sorted(
        set(int(k.split('.')[1]) for k in sd if 'weight' in k and len(sd[k].shape) == 2)
    )
    layer_sizes = [(sd[f'net.{i}.weight'].shape[1],
                    sd[f'net.{i}.weight'].shape[0]) for i in linear_keys]

    print(f'Detected architecture: {" → ".join(str(s[0]) for s in layer_sizes)} → {layer_sizes[-1][1]}')

    model = DeepfakeDetectorMLP(layer_sizes).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


# ── Load config ────────────────────────────────────────────────────────────
DEVICE = torch.device('cpu')

with open('config.json') as f:
    config = json.load(f)

THRESHOLD = config['threshold']
INPUT_DIM = config['input_dim']
FEAT_COLS = config['feat_cols']

# ── Load MLP (architecture auto-detected from checkpoint) ─────────────────
mlp_model, checkpoint = load_mlp_from_checkpoint('mlp_best.pt', DEVICE)
mlp_scaler = joblib.load('mlp_scaler.pkl')
print(f'✅ MLP loaded')

# ── Load Stacking ensemble ─────────────────────────────────────────────────
stacking_model = joblib.load('stacking.pkl')
print(f'✅ Stacking loaded | threshold={THRESHOLD:.4f}')
with open('config.json') as f:
    config = json.load(f)

THRESHOLD = config['threshold']
INPUT_DIM = config['input_dim']
FEAT_COLS = config['feat_cols']

# ── Load MLP ───────────────────────────────────────────────────────────────
checkpoint = torch.load('mlp_best.pt', map_location=DEVICE, weights_only=False)
mlp_model  = DeepfakeDetectorMLP(input_dim=INPUT_DIM).to(DEVICE)
mlp_model.load_state_dict(checkpoint['model_state'])
mlp_model.eval()
mlp_scaler = joblib.load('mlp_scaler.pkl')

# ── Load Stacking ensemble ─────────────────────────────────────────────────
stacking_model = joblib.load('stacking.pkl')

print(f'✅ Models loaded | threshold={THRESHOLD:.4f} | input_dim={INPUT_DIM}')


# ── Feature extraction ─────────────────────────────────────────────────────
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs.T, axis=0)

    lfccs     = compute_lfcc(y, fs=sr, num_ceps=20)
    lfcc_mean = np.mean(lfccs, axis=0)

    if FEAT_COLS[0].startswith('mfcc'):
        features = np.concatenate((mfcc_mean, lfcc_mean))
    else:
        features = np.concatenate((lfcc_mean, mfcc_mean))

    return features.reshape(1, -1)


# ── Prediction ─────────────────────────────────────────────────────────────
def predict(audio_path, selected_model):
    if audio_path is None:
        return '⚠️ Please upload an audio file.', '', ''

    try:
        features = extract_features(audio_path)

        if selected_model == 'MLP (Neural Network)':
            scaled = mlp_scaler.transform(features).astype(np.float32)
            with torch.no_grad():
                tensor = torch.tensor(scaled).to(DEVICE)
                prob   = torch.sigmoid(mlp_model(tensor)).item()

        elif selected_model == 'Stacking Ensemble':
            prob = stacking_model.predict_proba(features)[0][1]

        else:
            return 'Unknown model selected.', '', ''

        is_fake    = prob > THRESHOLD
        verdict    = '🔴 FAKE — AI Generated Voice' if is_fake else '🟢 REAL — Human Voice'
        confidence = prob * 100 if is_fake else (1 - prob) * 100

        details = f"""Fake probability : {prob*100:.1f}%
Real probability : {(1-prob)*100:.1f}%
Threshold used   : {THRESHOLD:.4f}
Model            : {selected_model}"""

        advice = (
            '⚠️ This audio shows patterns consistent with AI-generated or cloned voice.'
            if is_fake else
            '✅ This audio shows patterns consistent with a genuine human voice.'
        )

        return verdict, f'{confidence:.1f}% confident', details

    except Exception as e:
        return f'❌ Error: {e}', '', ''


# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title='🎙️ Deepfake Voice Detector') as demo:

    gr.Markdown("""
    # 🎙️ AI Voice Deepfake Detector
    Upload any audio file to detect whether it was generated by AI or recorded from a real human.
    Supports WAV, MP3, and FLAC formats.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input  = gr.Audio(type='filepath', label='🎵 Upload Audio File')
            model_choice = gr.Dropdown(
                choices=['Stacking Ensemble', 'MLP (Neural Network)'],
                value='Stacking Ensemble',
                label='🤖 Select Model'
            )
            run_btn = gr.Button('🔍 Analyze Audio', variant='primary', size='lg')

        with gr.Column(scale=1):
            verdict_box    = gr.Textbox(label='Verdict',     lines=1, interactive=False)
            confidence_box = gr.Textbox(label='Confidence',  lines=1, interactive=False)
            details_box    = gr.Textbox(label='Details',     lines=5, interactive=False)

    run_btn.click(
        fn=predict,
        inputs=[audio_input, model_choice],
        outputs=[verdict_box, confidence_box, details_box]
    )

    gr.Markdown("""
    ---
    **How it works:** Extracts MFCC + LFCC acoustic features and runs them through
    a trained stacking ensemble (LightGBM + XGBoost + Random Forest) or MLP neural network.
    """)

demo.launch()