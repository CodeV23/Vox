import os
import time
import argparse
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import numpy as np


# --------------------- RECORD FUNCTION ---------------------
def record_clip(filename: str, duration: float, sr: int = 16000, device=None):
    """Record a single clip from the microphone."""
    print(f"🎙️  Recording {duration:.1f}s → {filename}")
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype="int16", device=device)
    sd.wait()
    sf.write(filename, audio, sr)
    print("✅  Saved:", filename)


# --------------------- AUDIO CHUNKING ---------------------
def chunk_audio_file(input_path: str, output_dir: str, chunk_length_s: float = 1.0):
    """Split an audio file into smaller chunks (default 1 s)."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"✂️  Chunking {input_path} into {chunk_length_s}s pieces…")

    audio, sr = sf.read(input_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    chunk_samples = int(chunk_length_s * sr)
    chunks = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]

    base = os.path.splitext(os.path.basename(input_path))[0]
    for i, chunk in enumerate(chunks):
        if len(chunk) > 0:
            out_name = os.path.join(output_dir, f"{base}_{i:03d}.wav")
            sf.write(out_name, chunk, sr)

    print(f"✅  Saved {len(chunks)} chunks to {output_dir}\n")


# --------------------- SAFE REMOVE ---------------------
def safe_remove(path):
    """Quietly remove a file or directory if it exists."""
    try:
        if os.path.isdir(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    os.remove(fp)
            os.rmdir(path)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# --------------------- POSITIVE (WAKEWORD) ---------------------
def interactive_mode(save_dir: str, duration: float, sr: int = 16000,
                     count: int = 10, wakeword="Vox", device=None):

    os.makedirs(save_dir, exist_ok=True)

    existing = [f for f in os.listdir(save_dir)
                if f.endswith(".wav") and f.startswith(wakeword.lower())]

    next_index = 0
    if existing:
        nums = []
        for f in existing:
            try:
                nums.append(int(f.split("_")[1].split(".")[0]))
            except Exception:
                continue
        if nums:
            next_index = max(nums) + 1

    print(f"\n🎧 Interactive POSITIVE mode — say '{wakeword}' when prompted.")
    print(f"Duration: {duration}s | Count: {count} | Saving to: {save_dir}")
    print(f"Starting index: {next_index}\n")

    for i in range(next_index, next_index + count):
        input(f"Press [Enter] to record POS sample {i + 1 - next_index}/{count}...")
        filename = os.path.join(save_dir, f"{wakeword.lower()}_{i:03d}.wav")
        time.sleep(0.2)
        record_clip(filename, duration, sr, device)
        time.sleep(0.3)

    print(f"\n✅ Done recording {count} wake-word samples.\n")


# --------------------- INTERACTIVE NEGATIVES ---------------------
def interactive_negative_mode(save_dir: str, duration: float,
                              sr: int = 16000, count: int = 20, device=None):

    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.endswith(".wav")]

    next_index = len(existing)

    print(f"\n👻 Interactive NEGATIVE mode — say false trigger words")
    print(f"Duration: {duration}s | Count: {count} | Saving to: {save_dir}")
    print("Examples: wall, fall, ball, box, stocks, cocks, rocks\n")

    for i in range(next_index, next_index + count):
        input(f"Press [Enter] to record NEG sample {i + 1 - next_index}/{count}...")
        filename = os.path.join(save_dir, f"neg_{i:04d}.wav")
        time.sleep(0.2)
        record_clip(filename, duration, sr, device)
        time.sleep(0.3)

    print(f"\n✅ Done recording {count} NEGATIVE speech samples.\n")


# --------------------- NEGATIVE BACKGROUND ---------------------
def continuous_mode(save_dir: str, duration: float | None,
                    sr: int = 16000, device=None,
                    chunk_len: float = 1.0, keep_raw=False):

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🌙 Continuous background recording → {save_dir}")
    print("Press Ctrl+C to stop.\n")

    raw_dir = os.path.join(save_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = os.path.join(raw_dir, f"background_{start_time}.wav")

    try:
        if duration is None:
            print("(∞) Recording indefinitely... Ctrl+C to stop")
            while True:
                chunk_file = os.path.join(raw_dir, f"bg_{int(time.time())}.wav")
                record_clip(chunk_file, 10, sr, device)
                chunk_audio_file(chunk_file, save_dir, chunk_len)
                if not keep_raw:
                    safe_remove(chunk_file)
        else:
            record_clip(raw_file, duration, sr, device)
            chunk_audio_file(raw_file, save_dir, chunk_len)
            if not keep_raw:
                safe_remove(raw_file)

    except KeyboardInterrupt:
        print("\n🟥 Interrupted by user.")
    except Exception as e:
        print("❌ Error:", e)
    finally:
        if not keep_raw:
            safe_remove(raw_dir)

        print(f"\n💾 Finished background capture at: {save_dir}\n")


# --------------------- MAIN ENTRY ---------------------
def main():
    parser = argparse.ArgumentParser(description="Wake-word data recording tool")

    parser.add_argument("--mode",
                        choices=["pos", "neg", "neg_press"],
                        required=True,
                        help="pos = wakeword | neg = background | neg_press = spoken false positives")

    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--duration", type=str, default="1.0",
                        help="Clip length or 'None' for infinite background")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--wakeword", type=str, default="Vox")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--chunk_len", type=float, default=1.0)
    parser.add_argument("--keep_raw", action="store_true")

    args = parser.parse_args()

    # Handle None duration
    if args.duration.lower() == "none":
        args.duration = None
    else:
        args.duration = float(args.duration)

    if args.mode == "pos":
        save_dir = os.path.join(args.root, "pos")
        interactive_mode(save_dir, args.duration, args.sr,
                         args.count, args.wakeword, args.device)

    elif args.mode == "neg_press":
        save_dir = os.path.join(args.root, "neg_peoples")
        interactive_negative_mode(save_dir, args.duration,
                                  args.sr, args.count, args.device)

    else:
        save_dir = os.path.join(args.root, "neg")
        continuous_mode(save_dir, args.duration, args.sr,
                        args.device, args.chunk_len, args.keep_raw)


if __name__ == "__main__":
    main()


"""
🧠 ======================================================
         VOX RECORDING COMMAND CHEAT-SHEET
======================================================

🟢 Wakeword (True Positives):
python RecordVox.py --mode pos --count 40 --duration 1.0

🟡 Spoken False Positives (Press-Enter Mode):
python RecordVox.py --mode neg_press --count 40 --duration 1.0

🔴 Continuous Background Noise:
python RecordVox.py --mode neg --duration None --chunk_len 1.0

======================================================
"""
