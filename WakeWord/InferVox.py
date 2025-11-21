import torch, soundfile as sf, numpy as np, librosa

# ---------------- CONFIG ----------------
MODEL_PATH = "voxcrnn_frozen.pt"
SR = 16000
N_MELS = 40
HOP_LENGTH = 160
N_FFT = 512
WIN_LENGTH = 400
FIXED_FRAMES = 98

# ---------------- LOAD MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
print(f"🥹 Loaded model on {device}")

# ---------------- FEATURE FUNCTION ----------------
def extract_logmel(path):
    y, sr = sf.read(path)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))
    elif len(y) > sr * 1.1:
        y = y[:sr]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS, fmin=20, fmax=sr // 2
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # pad/crop to fixed frame count
    if logmel.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="constant")
    elif logmel.shape[1] > FIXED_FRAMES:
        logmel = logmel[:, :FIXED_FRAMES]

    # 🧠 Apply SAME normalization as training
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)

    # shape: [1, 1, 40, 98]
    return torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


# ---------------- INFERENCE ----------------
def infer(path, threshold=0.5):
    x = extract_logmel(path)
    with torch.no_grad():
        prob = model(x).item()
    print(f"🎧 {path}: {prob:.3f}")
    if prob > threshold:
        print("✅ Wake word detected!")
    else:
        print("❌ No wake word.")
    return prob

# Example usage:
infer(r"C:\code\python\Vox\WakeWord\data\pos\vox_256.wav")

infer(r"C:\code\python\Vox\WakeWord\data\neg\background_20251111_202225_1762910545_000.wav")

infer(r"C:\code\python\Vox\WakeWord\data\neg_peoples\ps_85153_000.wav")
