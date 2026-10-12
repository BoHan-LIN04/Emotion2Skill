"""Confidence-gated emotion context and skill selection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .encoder import EmotionSummary


@dataclass
class Skill:

    name:        str
    sop:         str
    description: str


@dataclass
class SkillLibrary:

    skills: list[Skill] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


    def __len__(self) -> int:
        return len(self.skills)

    def __getitem__(self, name: str) -> Skill:
        for s in self.skills:
            if s.name == name:
                return s
        raise KeyError(f"Skill '{name}' not found in library.")

    def names(self) -> list[str]:
        return [s.name for s in self.skills]

    def replace(self, name: str, new_skill: Skill) -> None:
        for i, s in enumerate(self.skills):
            if s.name == name:
                self.skills[i] = new_skill
                return
        raise KeyError(f"Skill '{name}' not found.")


    def format_menu(self) -> str:

        lines = ["Available skills:"]
        for i, s in enumerate(self.skills, 1):
            lines.append(f"  {i}. {s.name}: {s.description}")
        return "\n".join(lines)

    def format_sops(self, names: Optional[list[str]] = None) -> str:

        targets = names if names is not None else self.names()
        parts = []
        for name in targets:
            s = self[name]
            parts.append(f"=== Skill: {s.name} ===\n{s.sop}")
        return "\n\n".join(parts)


SELECTION_SYSTEM = """\
You are an autonomous agent equipped with a skill library. \
At each step you must choose exactly one skill to execute. \
Read the task context, available skills, and (when provided) the agent's \
internal emotion state, then output the name of the chosen skill on a single \
line — nothing else."""

SELECTION_USER_TEMPLATE = """\
## Task Context
{context}

## Available Skills
{skill_menu}

## Skill SOPs
{skill_sops}
{emotion_block}
## Decision
Respond with the skill name only (e.g. "ProductSearch").\
"""

EMOTION_BLOCK_TEMPLATE = """\

## Agent Internal State
{emotion_str}

"""


def build_selection_prompt(
    context:        str,
    library:        SkillLibrary,
    emotion:        Optional[EmotionSummary],
) -> str:
    """Build a routing prompt with admitted emotion context."""
    emotion_block = ""
    if emotion is not None and emotion.is_admitted:
        emotion_block = EMOTION_BLOCK_TEMPLATE.format(
            emotion_str=emotion.to_prompt_str()
        )

    user_msg = SELECTION_USER_TEMPLATE.format(
        context     = context.strip(),
        skill_menu  = library.format_menu(),
        skill_sops  = library.format_sops(),
        emotion_block = emotion_block,
    )
    return user_msg


@dataclass
class SelectionResult:
    skill_name:    str
    raw_output:    str
    prompt:        str
    emotion_used:  bool
    candidate_scores: dict[str, float] = field(default_factory=dict)


class SkillSelector:
    """Score valid identifiers, or parse an exact name/menu number from a text-only callable."""

    def __init__(self, llm_callable):
        self._llm = llm_callable

    def select(
        self,
        context:  str,
        library:  SkillLibrary,
        emotion:  Optional[EmotionSummary] = None,
    ) -> SelectionResult:

        prompt = build_selection_prompt(context, library, emotion)
        names = library.names()
        if not names:
            raise ValueError("Cannot select from an empty skill library.")
        scores: dict[str, float] = {}
        if callable(getattr(self._llm, "score_candidates", None)):
            values = self._llm.score_candidates(SELECTION_SYSTEM, prompt, names)
            scores = dict(values) if isinstance(values, dict) else dict(zip(names, values))
            if set(scores) != set(names):
                raise ValueError("Candidate scorer must return one score per valid skill.")
            import math
            if any(not math.isfinite(float(value)) for value in scores.values()):
                raise ValueError("Candidate scorer returned a non-finite score.")
            skill_name = max(names, key=lambda name: float(scores[name]))
            raw = skill_name
        else:
            raw = self._llm(SELECTION_SYSTEM, prompt).strip()
            skill_name = self._parse_skill(raw, library)

        return SelectionResult(
            skill_name   = skill_name,
            raw_output   = raw,
            prompt       = prompt,
            emotion_used = emotion is not None and emotion.is_admitted,
            candidate_scores = scores,
        )

    @staticmethod
    def _parse_skill(text: str, library: SkillLibrary) -> str:
        """Accept an exact skill name (case-insensitive) or a 1-based menu number."""
        names = library.names()

        answer = text.strip().strip('"\'').strip()
        if re.fullmatch(r"\d+", answer):
            index = int(answer) - 1
            if 0 <= index < len(names):
                return names[index]
        for name in names:
            if answer.casefold() == name.casefold():
                return name
        raise ValueError(f"Invalid skill selection {text!r}; expected one of {names} or its menu number.")
