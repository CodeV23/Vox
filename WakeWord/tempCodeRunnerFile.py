import torch, sounddevice as sd, numpy as np, librosa, time, collections, sys

# ---------------- CONFIG ----------------
MODEL_PATH = "voxcrnn_frozen.pt"
SR = 16000
N_MELS = 40
HOP_LENGTH = 160
N_FFT = 512
WIN_LENGTH = 400
FIXED_FRAMES = 98

CHUNK_SEC = 1.0       # analysis window size
OVERLAP = 0.5         # overlap between windows (in seconds)
SMOOTH_WINDOW = 1     # smoother average, fewer lags
TRIGGER_THRESHOLD = 0.5
DEBOUNCE_SEC = 3.5

# ---------------- LOAD MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
print(f"✅ Loaded TorchScript model on {device}")

# ---------------- FEATURE EXTRACTION ----------------
def extract_logmel(y):
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    if len(y) < SR:
        y = np.pad(y, (0, SR - len(y)))
    elif len(y) > SR * 1.1:
        y = y[:SR]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS, fmin=20, fmax=SR//2
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # pad/crop to fixed length
    if logmel.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - logmel.shape[1]
        logmel = np.pad(logmel, ((0,0),(0,pad)), mode="constant")
    elif logmel.shape[1] > FIXED_FRAMES:
        logmel = logmel[:, :FIXED_FRAMES]

    # normalize per sample
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
    x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return x

# ---------------- LIVE DETECTION ----------------
def listen_and_detect():
    print("\n🎙️ Listening for wake word... (Ctrl+C to stop)")
    preds = collections.deque(maxlen=SMOOTH_WINDOW)
    last_trigger = 0

    try:
        blocksize = int(SR * CHUNK_SEC)
        hop = int(SR * (CHUNK_SEC - OVERLAP))
        buffer = np.zeros(blocksize, dtype=np.float32)

        with sd.InputStream(channels=1, samplerate=SR, blocksize=hop, dtype='float32') as stream:
            # --- warm-up to clear mic static ---
            print("🎧 Warming up mic...")
            for _ in range(3):
                stream.read(hop)
            print("✅ Warm-up complete, starting detection!\n")

            while True:
                new_audio, overflow = stream.read(hop)
                if overflow:
                    print("⚠️ Overflow!")

                # update rolling buffer (overlapping windows)
                buffer = np.roll(buffer, -hop)
                buffer[-hop:] = np.squeeze(new_audio)
                buffer = buffer / (np.max(np.abs(buffer)) + 1e-6)

                x = extract_logmel(buffer)
                with torch.no_grad():
                    p = model(x).item()

                preds.append(p)
                avg_p = np.mean(preds)

                bar = "█" * int(avg_p * 20)
                sys.stdout.write(f"\r[{bar:<20}] raw={p:.3f} avg={avg_p:.3f}")
                sys.stdout.flush()

                # trigger detection
                if avg_p > TRIGGER_THRESHOLD and time.time() - last_trigger > DEBOUNCE_SEC:
                    print("\n✅ Wake word detected! 🔊")
                    last_trigger = time.time()

                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🟥 Stopped listening.")
    except Exception as e:
        print("❌ Error:", e)

# ---------------- RUN ----------------
if __name__ == "__main__":
    listen_and_detect()
