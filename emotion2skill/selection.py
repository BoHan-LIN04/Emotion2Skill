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
from typing import Optional

import torch

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


class SkillSelector:
    """
    Emotion-Augmented Skill Selection (Eq. 9–11).

    Calls the LLM with the augmented prompt and parses the output to
    identify the chosen skill name.  Falls back to the first skill if
    parsing fails.

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
        raw    = self._llm(SELECTION_SYSTEM, prompt).strip()

        # Parse: find first matching skill name in model output
        skill_name = self._parse_skill(raw, library)

        return SelectionResult(
            skill_name   = skill_name,
            raw_output   = raw,
            prompt       = prompt,
            emotion_used = emotion is not None and emotion.is_admitted,
        )

    @staticmethod
    def _parse_skill(text: str, library: SkillLibrary) -> str:
        """
        Extract the skill name from model output.
        Tries exact match, then case-insensitive, then partial substring.
        Falls back to the first skill if nothing matches.
        """
        names = library.names()

        # 1. Exact match anywhere in the output
        for name in names:
            if name in text:
                return name

        # 2. Case-insensitive
        text_lower = text.lower()
        for name in names:
            if name.lower() in text_lower:
                return name

        # 3. Partial word match (e.g. "product" → "ProductSearch")
        words = re.findall(r"\w+", text_lower)
        for word in words:
            for name in names:
                if word in name.lower() and len(word) >= 4:
                    return name

        # Fallback
        return names[0]
