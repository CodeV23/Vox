import os, json, random, numpy as np, soundfile as sf

def rms_db(audio):
    """Compute RMS in dB."""
    rms = np.sqrt(np.mean(np.square(audio)))
    return 20 * np.log10(rms + 1e-9)

def normalize(audio):
    """Normalize amplitude to -30 dB RMS target."""
    current = rms_db(audio)
    gain = -30 - current
    factor = 10 ** (gain / 20)
    return np.clip(audio * factor, -1.0, 1.0)

def collect_samples(root_dir, label):
    """Collect .wav samples and filter out silent ones."""
    samples = []
    for file in os.listdir(root_dir):
        if not file.endswith(".wav"): continue
        path = os.path.join(root_dir, file)
        try:
            audio, sr = sf.read(path)
            if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
            if len(audio) < sr * 0.5:  # too short
                continue
            if rms_db(audio) < -45:  # too silent
                continue
            audio = normalize(audio)
            sf.write(path, audio, sr)  # overwrite normalized
            samples.append({"path": path, "label": label})
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
    return samples

def main():
    base = "data"
    random.seed(42)

    pos = collect_samples(os.path.join(base, "pos"), 1)
    neg_amb = collect_samples(os.path.join(base, "neg"), 0)
    neg_sp = collect_samples(os.path.join(base, "neg_peoples"), 0)
    print(f"✅ Collected: {len(pos)} pos, {len(neg_amb)} neg_amb, {len(neg_sp)} neg_sp")


    # ✅ split per class (prevents train/val overlap)
    split_pos = int(0.8 * len(pos))
    split_amb = int(0.8 * len(neg_amb))
    split_sp = int(0.8 * len(neg_sp))

    train = pos[:split_pos] + neg_amb[:split_amb] + neg_sp[:split_sp]
    val = pos[split_pos:] + neg_amb[split_amb:] + neg_sp[split_sp:]

    random.shuffle(train)
    random.shuffle(val)

    with open("train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open("val.json", "w") as f:
        json.dump(val, f, indent=2)

    print(f"✅ Saved {len(train)} train / {len(val)} val samples.")
    if train:
        print("Example entry:", train[0])

if __name__ == "__main__":
    main()
