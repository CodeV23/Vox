import os, json, numpy as np, soundfile as sf, librosa, random

# -------------------- AUGMENTATION --------------------
def augment_wave(y, sr):
    """Apply mild, safe audio augmentation."""
    # Light gain
    y = y * random.uniform(0.9, 1.1)

    # Small pitch shift ±0.5 semitone
    n_steps = random.uniform(-0.5, 0.5)
    y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

    # Small time stretch 0.95–1.05×
    rate = random.uniform(0.95, 1.05)
    y = librosa.effects.time_stretch(y=y, rate=rate)

    # Fix length = 1 sec
    if len(y) > sr:
        y = y[:sr]
    else:
        y = np.pad(y, (0, sr - len(y)))

    # Light Gaussian noise
    y = y + 0.002 * np.random.randn(len(y))
    return np.clip(y, -1.0, 1.0)

# -------------------- FEATURE EXTRACTION --------------------
def extract_logmel(y, sr=16000, n_mels=40, n_fft=512, hop_length=160,
                   win_length=400, fixed_frames=98):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=20, fmax=sr // 2
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # pad or crop to fixed frames
    if logmel.shape[1] < fixed_frames:
        pad = fixed_frames - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="constant")
    elif logmel.shape[1] > fixed_frames:
        logmel = logmel[:, :fixed_frames]

    # normalize per-sample (not globally)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
    return logmel.astype(np.float32)

def load_manifest(json_path):
    with open(json_path) as f:
        return json.load(f)

# -------------------- MAIN BUILDER --------------------
def build_features(json_path, out_path, oversample_factor=2):
    """
    Build NPZ features with optional augmentation for positive samples.
    Re-running will overwrite the file (no infinite duplication).
    """
    data = load_manifest(json_path)
    X, y = [], []
    print(f"📦 Loading {len(data)} entries from {json_path}")

    for i, sample in enumerate(data):
        try:
            y_raw, sr = sf.read(sample["path"])
            if len(y_raw.shape) > 1:
                y_raw = np.mean(y_raw, axis=1)
            if len(y_raw) < sr:
                y_raw = np.pad(y_raw, (0, sr - len(y_raw)))
            elif len(y_raw) > sr * 1.1:
                y_raw = y_raw[:sr]

            # Base feature
            feat = extract_logmel(y_raw, sr)

            # Optional mild augmentation for negatives (simulate mic pickup)
            if sample["label"] == 0:
                # random EQ-like tilt and noise
                if random.random() < 0.5:
                    tilt = np.linspace(1.0, random.uniform(0.6, 1.0), len(y_raw))
                    y_raw = y_raw * tilt
                if random.random() < 0.5:
                    y_raw = y_raw + 0.01 * np.random.randn(len(y_raw))
                    
            X.append(feat)
            y.append(sample["label"])

            # Positive augmentations
            if sample["label"] == 1 and oversample_factor > 0:
                for _ in range(oversample_factor):
                    aug = augment_wave(y_raw.copy(), sr)
                    feat_aug = extract_logmel(aug, sr)
                    X.append(feat_aug)
                    y.append(1)

        except Exception as e:
            print(f"⚠️ {sample['path']}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(data)}")

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"✅ Saved {out_path}: {X.shape[0]} samples ({sum(y)} pos / {len(y)-sum(y)} neg)")

# -------------------- RUN --------------------
def main():
    print("🎛️ Extracting training features with augmentation...")
    build_features("train.json", "features_train.npz", oversample_factor=2)

    print("\n🎚️ Extracting validation features (no augmentation)...")
    build_features("val.json", "features_val.npz", oversample_factor=0)

if __name__ == "__main__":
    main()
