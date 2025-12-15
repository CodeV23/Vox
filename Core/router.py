import ollama
from Agents.Music.music_agent import MusicAgent
from Agents.Chef.chef_agent import ChefAgent


class Router:
    def __init__(self):
        self.agent_map = {
            "music": MusicAgent(),
            "chef": ChefAgent(),
        }

    def classify(self, text: str) -> str:
        response = ollama.chat(
            model="phi3:mini",
            options={"num_predict": 1},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Route the request:\n"
                        "- music: songs, artists, playlists, listening\n"
                        "- chef: cooking, food, recipes\n"
                        "- research: research, brainstorming, writing\n"
                        "- none: anything else\n"
                        "Reply with ONE WORD."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )

        label = response["message"]["content"].strip().lower()
        return label if label in self.agent_map else "none"

    def route(self, text: str):
        return self.agent_map.get(self.classify(text))
