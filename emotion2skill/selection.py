"""
Emotion-Augmented Skill Selection  (§3.3)
==========================================

Implements Eqs. (9)–(11) of the paper:

  P_t = [x_t; {D_i}; T_t*; c_t]  if c_t >= τ
        [x_t; {D_i}]              otherwise

  s_t* = argmax_{s_i ∈ S}  p_M(s_i | P_t)

The LLM receives the augmented prompt and selects a skill by constrained
generation over valid skill identifiers.  Setting τ=1 recovers the
standard text-only selection baseline (MASA-style).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .encoder import EmotionSummary


# ─────────────────────────────────────────────────────────────────────────────
# Skill data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    """
    A single skill in the library S.

    Attributes
    ----------
    name        : short identifier used in prompt (e.g. "ProductSearch")
    sop         : full natural-language Standard Operating Procedure D_i
    description : one-line summary for the selection menu
    """
    name:        str
    sop:         str
    description: str


@dataclass
class SkillLibrary:
    """
    The skill library S = {s_1, ..., s_M}.
    """
    skills: list[Skill] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ── accessors ─────────────────────────────────────────────────────────────

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

    # ── prompt formatting ─────────────────────────────────────────────────────

    def format_menu(self) -> str:
        """Format the skill selection menu (name + one-line description)."""
        lines = ["Available skills:"]
        for i, s in enumerate(self.skills, 1):
            lines.append(f"  {i}. {s.name}: {s.description}")
        return "\n".join(lines)

    def format_sops(self, names: Optional[list[str]] = None) -> str:
        """Format the full SOPs for the listed skills (or all)."""
        targets = names if names is not None else self.names()
        parts = []
        for name in targets:
            s = self[name]
            parts.append(f"=== Skill: {s.name} ===\n{s.sop}")
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    Constructs P_t (Eq. 9) for the skill-selection call.

    Parameters
    ----------
    context  : x_t — concatenated task instruction + history + observation
    library  : S
    emotion  : φ_t from the encoder; if None or not admitted, omitted

    Returns
    -------
    Full prompt string passed to the LLM.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Skill selector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SelectionResult:
    skill_name:    str
    raw_output:    str
    prompt:        str
    emotion_used:  bool
    candidate_scores: dict[str, float] = field(default_factory=dict)


class SkillSelector:
    """
    Emotion-Augmented Skill Selection (Eq. 9–11).

    Scores every valid identifier by its conditional token log probability.
    Text-only mock callables must return one exact identifier or menu number;
    ambiguous output is rejected instead of silently selecting the first skill.

    Parameters
    ----------
    llm_callable : callable(system: str, user: str) -> str
        Any function that accepts system + user strings and returns
        the LLM's text completion.  Wired to the Qwen3 wrapper in agent.py.
    """

    def __init__(self, llm_callable):
        self._llm = llm_callable

    def select(
        self,
        context:  str,
        library:  SkillLibrary,
        emotion:  Optional[EmotionSummary] = None,
    ) -> SelectionResult:
        """
        Perform one skill-selection step.

        Returns
        -------
        SelectionResult with the chosen skill name and metadata.
        """
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
        """
        Extract the skill name from model output.
        Accept an entire exact skill name (case-insensitive) or a 1-based
        menu number. Never interpret a mention in explanatory prose as a vote.
        """
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
