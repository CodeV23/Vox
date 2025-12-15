import ollama
from Agents.Research.tools import ResearchTools


class ResearchAgent:
    def __init__(
        self,
        planner_model: str = "llama3.1:8b",
        writer_model: str = "llama3.1:8b",
        critic_model: str = "llama3.2:3b",
        max_notes_per_turn: int = 10,
        max_history_turns: int = 12,
    ):
        self.tools = ResearchTools()
        self.planner_model = planner_model
        self.writer_model = writer_model
        self.critic_model = critic_model
        self.max_notes_per_turn = max_notes_per_turn
        self.max_history_turns = max_history_turns

        # Persistent conversation context
        self.chat_history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]

        self.persona = (
            "You are VoxResearch, a precise brainstorming + research agent.\n"
            "You keep context across turns. Ask clarifying questions only if truly necessary.\n"
            "Avoid hallucinations: if uncertain, state assumptions.\n"
            "When helpful, structure answers with bullets and next actions.\n"
        )

    def _chat(self, model: str, system: str, user: str, num_predict: int = 900):
        # Include recent history for continuity (big context, bounded)
        history = self.chat_history[-2 * self.max_history_turns :]  # user+assistant pairs
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]

        resp = ollama.chat(
            model=model,
            options={"num_predict": num_predict, "temperature": 0.4},
            messages=messages,
        )
        return (resp["message"]["content"] or "").strip()

    def _compress_summary(self, raw: str) -> str:
        sys = (
            "You are a summarizer. Produce a compact durable memory summary.\n"
            "Include: objective, constraints, key facts, decisions, open questions.\n"
            "Max 2500 characters."
        )
        out = self._chat(self.critic_model, sys, raw, num_predict=500)
        return out[:2500]

    def _maybe_reset(self, user_text: str) -> bool:
        t = (user_text or "").strip().lower()
        if t in {"reset", "clear", "new topic"}:
            self.tools.reset()
            self.chat_history.clear()
            print("VoxResearch: Reset done. New topic—hit me.")
            return True
        return False

    def handle(self, user_text: str):
        if not user_text:
            # Called by "switch research" etc. Don’t do anything.
            return ""

        if self._maybe_reset(user_text):
            return ""

        # Update goal only if empty; otherwise keep current goal for continuity
        if not self.tools.ws.goal:
            self.tools.set_goal(user_text)

        # ---- STAGE 1: PLAN (lightweight, only if we don't have one yet or user asked for plan) ----
        want_plan = any(k in user_text.lower() for k in ["plan", "roadmap", "steps", "approach"])
        if not self.tools.ws.plan or want_plan:
            plan_prompt = f"""
CURRENT GOAL:
{self.tools.ws.goal}

USER MESSAGE:
{user_text}

Return a short plan (5–10 steps). Keep it actionable.
"""
            plan = self._chat(self.planner_model, self.persona, plan_prompt, num_predict=550)
            self.tools.save_plan(plan)

        # ---- STAGE 2: NOTES (incremental per turn) ----
        notes_prompt = f"""
GOAL:
{self.tools.ws.goal}

PLAN:
{self.tools.ws.plan}

USER MESSAGE:
{user_text}

Generate up to {self.max_notes_per_turn} incremental research notes:
- bullets
- cover angles/tradeoffs/examples
- include assumptions/unknowns if relevant
Only return the bullets.
"""
        notes = self._chat(self.writer_model, self.persona, notes_prompt, num_predict=800)
        added = 0
        for line in notes.splitlines():
            line = line.strip().lstrip("-• \t").strip()
            if line:
                self.tools.add_note(line)
                added += 1
            if added >= self.max_notes_per_turn:
                break

        # ---- STAGE 3: SYNTHESIS ----
        snapshot = self.tools.snapshot()
        draft_prompt = f"""
Use the workspace + conversation context to answer the USER MESSAGE well.

WORKSPACE:
Goal: {snapshot["goal"]}

Plan:
{snapshot["plan"]}

Recent notes:
- """ + "\n- ".join(snapshot["notes"]) + f"""

USER MESSAGE:
{user_text}

Write a strong response:
- lead with the best answer
- give rationale + options
- end with next actions
No fluff.
"""
        final = self._chat(self.writer_model, self.persona, draft_prompt, num_predict=1200)

        # ---- Rolling summary (optional) ----
        memory_blob = (
            f"GOAL:\n{self.tools.ws.goal}\n\nPLAN:\n{self.tools.ws.plan}\n\nNOTES:\n"
            + "\n".join(self.tools.ws.notes[-50:])
        )
        self.tools.ws.rolling_summary = self._compress_summary(memory_blob)

        # Persist conversation history for next turn
        self.chat_history.append({"role": "user", "content": user_text})
        self.chat_history.append({"role": "assistant", "content": final})

        print(final)
        return final
