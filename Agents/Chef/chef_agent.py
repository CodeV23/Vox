import ollama

class ChefAgent:
    def __init__(self):
        self.persona = (
            "You are Voxel-Chef — energetic, friendly, practical. "
            "Give ingredients, cooking steps, tips, and substitutions. "
            "Be concise and helpful."
        )

    def handle(self, text):
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {"role": "system", "content": self.persona},
                {"role": "user", "content": text}
            ]
        )
        return response["message"]["content"].strip()
