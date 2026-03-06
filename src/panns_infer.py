import torch
import librosa
import numpy as np
from models.models import Cnn14
import soundfile as sf
import os

# -------- CONFIG --------
CHECKPOINT_PATH = "panns/Cnn14_16k_mAP=0.438.pth"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------- LOAD MODEL --------
model = Cnn14(
    sample_rate=16000,
    window_size=512,
    hop_size=160,
    mel_bins=64,
    fmin=50,
    fmax=8000,
    classes_num=527
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()


# -------- AUDIO → SOUND EVENTS --------
def detect_sound_events(audio_path, top_k=5):
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    waveform = torch.tensor(waveform).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(waveform)
        probs = output["clipwise_output"].cpu().numpy()[0]

    # Load class labels
    LABEL_PATH = os.path.join(os.path.dirname(__file__), "class_labels_indices.csv")
    with open(LABEL_PATH) as f:

        labels = [line.strip().split(",")[2] for line in f.readlines()[1:]]

    top_indices = np.argsort(probs)[-top_k:][::-1]
    results = [(labels[i], float(probs[i])) for i in top_indices]

    return results


if __name__ == "__main__":
    events = detect_sound_events("audio.wav")
    print("\n===== SOUND EVENTS =====\n")
    for label, prob in events:
        print(f"{label}: {prob:.3f}")
