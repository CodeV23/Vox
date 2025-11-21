import ollama
from Agents.Music.music_agent import MusicAgent
from Agents.Chef.chef_agent import ChefAgent


class Router:
    def __init__(self):
        self.agent_map = {
            "music": MusicAgent(),
            "chef": ChefAgent(),
        }

        self.valid_labels = ", ".join(self.agent_map.keys()) + ", none"

    def classify(self, text: str) -> str:
        system_prompt = (
            "You are Voxel's routing brain.\n"
            f"Valid labels: {self.valid_labels}.\n"
            "Pick the correct label for the user's request.\n"
            "- If it's about music, playlists, songs, vibes → music\n"
            "- If it's about recipes, cooking, ingredients → chef\n"
            "- Otherwise → none\n"
            "Respond with ONE WORD ONLY."
        )

        response = ollama.chat(
            model="phi3:mini",               # SUPER FAST
            options={"num_predict": 1},      # classification only
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
        )

        raw = response["message"]["content"].strip().lower()
        label = raw.split()[0]

        if label not in self.agent_map and label != "none":
            label = "none"

        return label

    def route(self, text: str):
        label = self.classify(text)
        return self.agent_map.get(label, None)
