import ollama
from Core.router import Router

class BaseAgent:
    def __init__(self):
        self.router = Router()
        with open("Core/prompts/base_persona.txt", "r", encoding="utf-8") as f:
            self.base_personality = f.read()

    def respond(self, user_text: str):
        agent = self.router.route(user_text)
        agent_label = agent.__class__.__name__ if agent else "none"

        quip_prompt = (
            f"User said: {user_text}\n"
            f"Selected agent: {agent_label}\n"
            "You are Voxel Base (Jarvis-like). RULES:\n"
            "- DO NOT provide any domain content.\n"
            "- DO NOT recommend music, recipes, or anything task-specific.\n"
            "- DO NOT solve the user's problem.\n"
            "- ONLY produce a witty, dry, classy handoff line.\n"
            "- 1 or 2 sentences maximum.\n"
            "- You MUST end with a clear handoff phrase:\n"
            "  '—— Switching to {agent_label} ——'\n"
            "That's ALL you output."
        )

        print("\nVOXEL: ", end="", flush=True)

        stream = ollama.chat(
            model="llama3.2:3b",
            stream=True,
            options={"num_predict": 60},
            messages=[
                {"role": "system", "content": self.base_personality},
                {"role": "user", "content": quip_prompt}
            ]
        )

        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print()

        if agent is None:
            return ""

        # Tell DJ-V to start cleanly
        print("—— DJ-V TAKES OVER ——")
        agent.handle(user_text)
        return ""
