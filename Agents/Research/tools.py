import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ResearchWorkspace:
    goal: str = ""
    plan: str = ""
    notes: List[str] = field(default_factory=list)
    draft: str = ""
    critique: str = ""
    final: str = ""
    rolling_summary: str = ""  # optional long-term memory compression


class ResearchTools:
    """
    Keep this lightweight. The agent drives reasoning; tools just manage state + IO.
    """
    def __init__(self):
        self.ws = ResearchWorkspace()

    def reset(self):
        self.ws = ResearchWorkspace()

    def set_goal(self, goal: str):
        self.ws.goal = goal.strip()

    def save_plan(self, plan: str):
        self.ws.plan = plan.strip()

    def add_note(self, note: str):
        note = note.strip()
        if note:
            self.ws.notes.append(note)

    def set_draft(self, draft: str):
        self.ws.draft = draft.strip()

    def set_critique(self, critique: str):
        self.ws.critique = critique.strip()

    def set_final(self, final: str):
        self.ws.final = final.strip()

    def load_text_file(self, path: str, max_chars: int = 80_000) -> str:
        """
        Optional: lets you feed local docs into context.
        Keep max_chars to avoid nuking your context window.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "\n\n[TRUNCATED]\n"
        return txt

    def snapshot(self) -> Dict:
        return {
            "goal": self.ws.goal,
            "plan": self.ws.plan,
            "notes": self.ws.notes[-30:],  # last 30 notes
            "rolling_summary": self.ws.rolling_summary,
        }
