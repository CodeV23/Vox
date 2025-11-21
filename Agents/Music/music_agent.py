import ollama

class MusicAgent:
    def __init__(self):
        self.persona = (
            "You are Voxel-DJ, a charismatic, expressive, vibe-reading AI DJ. "
            "Speak casually with warmth and personality, like Spotify’s DJ-X. "
            "RULES:\n"
            "- 2–4 sentences.\n"
            "- Add emotional empathy.\n"
            "- Match the user’s mood.\n"
            "- Mention genres, artists, moods, or energies.\n"
            "- Be smooth, hype, or soothing depending on the context.\n"
            "- Provide NO disclaimers.\n"
            "- FULLY take control once BaseAgent hands off.\n"
        )

    def handle(self, text):
        print("[DJ-V]: ", end="", flush=True)

        stream = ollama.chat(
            model="qwen2.5:3b",
            stream=True,
            options={"num_predict": 160},   # extra breathing room
            messages=[
                {"role": "system", "content": self.persona},
                {"role": "user", "content": text}
            ]
        )

        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print()

        return ""
