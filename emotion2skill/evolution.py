"""
Emotion-Driven Skill Evolution  (§3.4)
=======================================

Implements Eqs. (12)–(17) of the paper.

Pipeline per evolution round r
--------------------------------
1. For every completed episode in the batch, compute step-to-step
   emotion shift magnitudes δ_t (cosine distance, Eq. 12).
2. Identify emotion transition points P (Eq. 13): steps where δ_t
   exceeds the episode mean by more than 1 std-dev.
3. For each underperforming skill s_i (success rate < η), restrict
   transition points to those where s_i was active → P_i.
4. Extract 3-step diagnostic windows d_t around each point (Eq. 14).
5. Verbalize transition patterns into Δ_i.
6. Build evolution prompt P_evo^(i) (Eq. 15) and call the LLM rewriter.
7. Replace s_i with the rewritten SOP s_i' in the library.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .encoder import EMOTION_TEMPLATES, EmotionSummary
from .selection import Skill, SkillLibrary


# ─────────────────────────────────────────────────────────────────────────────
# Episode log entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Everything recorded at a single decision step t."""
    step:            int
    observation:     str               # x_t
    skill_selected:  str               # s_t*  (skill name)
    template_idx:    int               # T_t*
    confidence:      float             # c_t
    emotion_state:   torch.Tensor      # e_t ∈ R^K (ℓ2-normalised)
    action_taken:    str               # actual environment action
    reward:          float             # immediate step reward (0 if N/A)


@dataclass
class EpisodeLog:
    """
    Complete log for one episode.

    Attributes
    ----------
    episode_id : str
    steps      : ordered list of StepRecord
    outcome    : float — episode-level reward / success (r_ep)
    success    : bool  — True if outcome >= success threshold
    """
    episode_id: str
    steps:      list[StepRecord] = field(default_factory=list)
    outcome:    float = 0.0
    success:    bool  = False

    def add_step(self, record: StepRecord) -> None:
        self.steps.append(record)

    def __len__(self) -> int:
        return len(self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# Transition-point detection (Eqs. 12–13)
# ─────────────────────────────────────────────────────────────────────────────

def compute_shift_magnitudes(episode: EpisodeLog) -> list[float]:
    """
    Eq. (12): δ_t = 1 - cosine_similarity(e_t, e_{t-1})
                  = cosine distance between consecutive emotion states.

    Returns
    -------
    deltas : list of length (T-1), indexed from step index 1..T-1
             (delta[0] corresponds to the shift from step 0 → step 1)
    """
    states = [s.emotion_state for s in episode.steps]
    if len(states) < 2:
        return []

    deltas: list[float] = []
    for t in range(1, len(states)):
        e_prev = F.normalize(states[t - 1].float(), dim=0)
        e_curr = F.normalize(states[t].float(),     dim=0)
        cos_sim = (e_prev * e_curr).sum().item()
        # Clamp to [-1, 1] for numerical safety
        cos_sim = max(-1.0, min(1.0, cos_sim))
        deltas.append(1.0 - cos_sim)
    return deltas


def find_transition_points(
    episode: EpisodeLog,
    deltas:  list[float],
) -> list[int]:
    """
    Eq. (13): P = { t ∈ {2,...,T} | δ_t > δ̄ + σ_δ }

    Note: step indices in the returned list are 1-based positions into
    episode.steps (so t=1 means the shift from steps[0]→steps[1]).

    Returns
    -------
    transition_steps : list of step indices (1-indexed into episode.steps)
    """
    if len(deltas) < 2:
        return []

    mean_d = statistics.mean(deltas)
    std_d  = statistics.pstdev(deltas)   # population std, as in paper

    return [
        t + 1   # convert 0-indexed delta → 1-indexed step
        for t, d in enumerate(deltas)
        if d > mean_d + std_d
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic context extraction (Eq. 14)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagnosticWindow:
    """
    d_t = (x_{t-1:t+1}, s*_{t-1:t+1}, T*_{t-1:t+1}, δ_{t-1:t+1})
    3-step window centred on transition step t.
    """
    center_step:    int
    observations:   list[str]       # x_{t-1}, x_t, x_{t+1}
    skills:         list[str]       # s*_{t-1}, s*_t, s*_{t+1}
    templates:      list[str]       # T*_{t-1}, T*_t, T*_{t+1}
    shift_mags:     list[float]     # δ_{t-1}, δ_t, δ_{t+1}   (padded with 0)
    skill_of_interest: str          # s_i being diagnosed

    def verbalize(self) -> str:
        """
        Produce the verbalized transition summary Δ_i for the rewriter prompt.
        """
        lines = [
            f"  [Transition at step {self.center_step}]",
        ]
        labels = ["t-1", "t", "t+1"]
        for k, lbl in enumerate(labels):
            lines.append(
                f"  Step {lbl}: obs='{self.observations[k][:80]}' | "
                f"skill={self.skills[k]} | "
                f"emotion={self.templates[k]} | "
                f"δ={self.shift_mags[k]:.3f}"
            )
        return "\n".join(lines)


def extract_diagnostic_windows(
    episode:    EpisodeLog,
    transitions: list[int],
    skill_name:  str,
    deltas:      list[float],
) -> list[DiagnosticWindow]:
    """
    For each transition step t where skill_name was active,
    extract the 3-step window d_t (Eq. 14).

    Parameters
    ----------
    episode     : full episode log
    transitions : skill-specific transition step indices (1-indexed)
    skill_name  : s_i
    deltas      : full δ list (0-indexed, length T-1)

    Returns
    -------
    windows : list of DiagnosticWindow
    """
    T = len(episode.steps)
    windows: list[DiagnosticWindow] = []

    # Filter to steps where this skill was active
    skill_transitions = [
        t for t in transitions
        if 0 <= t < T and episode.steps[t].skill_selected == skill_name
    ]

    for t in skill_transitions:
        # 3-step window: t-1, t, t+1 (clamped to [0, T-1])
        indices = [max(0, t - 1), t, min(T - 1, t + 1)]
        observations = [episode.steps[i].observation[:200] for i in indices]
        skills       = [episode.steps[i].skill_selected    for i in indices]
        templates    = [
            EMOTION_TEMPLATES[episode.steps[i].template_idx] for i in indices
        ]
        # δ values: delta[t-1] is the shift arriving at step t
        def _safe_delta(idx: int) -> float:
            # deltas is 0-indexed, deltas[i] = shift from step i to i+1
            di = idx - 1   # shift arriving at step idx
            return deltas[di] if 0 <= di < len(deltas) else 0.0

        shift_mags = [_safe_delta(i) for i in indices]

        windows.append(DiagnosticWindow(
            center_step       = t,
            observations      = observations,
            skills            = skills,
            templates         = templates,
            shift_mags        = shift_mags,
            skill_of_interest = skill_name,
        ))

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Evolution prompt builder (Eqs. 15–16)
# ─────────────────────────────────────────────────────────────────────────────

EVOLUTION_SYSTEM = """\
You are an expert agent trainer. \
Your task is to rewrite a skill SOP to fix the failure patterns \
identified in the emotion diagnostics below. \
Keep the SOP concise, actionable, and structured. \
Output ONLY the revised SOP text — no explanation, no preamble."""

EVOLUTION_USER_TEMPLATE = """\
## Skill Being Revised
Name: {skill_name}

## Current SOP
{current_sop}

## Episode Outcome
{outcome_str}

## Emotion Transition Diagnostics
The following transitions were detected during this episode while \
this skill was active. Each shows 3 steps centred on a sharp internal \
state change (high δ = the agent's emotion shifted abruptly):

{diagnostics_str}

## Transition Pattern Summary
{delta_summary}

## Instructions
Rewrite the SOP to address the failure modes revealed by the diagnostics. \
Focus on:
- Clarifying what the agent should do when confused or frustrated states arise.
- Adding explicit recovery steps for abrupt internal-state transitions.
- Preserving the parts of the SOP that work well.

Output the revised SOP now:\
"""


def build_evolution_prompt(
    skill:    Skill,
    windows:  list[DiagnosticWindow],
    outcome:  float,
    success:  bool,
) -> str:
    """
    Eq. (15): P_evo^(i) = [D_i; {d_t}_{t∈P_i}; Δ_i; r_ep]

    Returns the full user-turn string for the rewriter LLM.
    """
    outcome_str = (
        f"{'SUCCESS' if success else 'FAILURE'} (score = {outcome:.3f})"
    )

    if windows:
        diagnostics_str = "\n\n".join(w.verbalize() for w in windows)
        # Verbalized summary Δ_i
        n_transitions = len(windows)
        avg_shift = sum(w.shift_mags[1] for w in windows) / n_transitions
        dominant_templates = [w.templates[1] for w in windows]
        template_freq: dict[str, int] = {}
        for t in dominant_templates:
            template_freq[t] = template_freq.get(t, 0) + 1
        most_common = max(template_freq, key=template_freq.get)

        delta_summary = (
            f"Detected {n_transitions} abrupt emotion transition(s) while "
            f"skill '{skill.name}' was active. "
            f"Average shift magnitude at transitions: {avg_shift:.3f}. "
            f"Most frequent emotion state at transitions: '{most_common}'. "
            f"These indicate the agent encountered repeated decision uncertainty "
            f"or frustration during execution of this skill."
        )
    else:
        diagnostics_str = "(No skill-specific transitions detected.)"
        delta_summary = (
            "No abrupt emotion transitions were detected for this skill "
            "in this episode. Consider reviewing the overall SOP structure."
        )

    return EVOLUTION_USER_TEMPLATE.format(
        skill_name       = skill.name,
        current_sop      = skill.sop,
        outcome_str      = outcome_str,
        diagnostics_str  = diagnostics_str,
        delta_summary    = delta_summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Skill performance tracker
# ─────────────────────────────────────────────────────────────────────────────

class SkillPerformanceTracker:
    """
    Tracks per-skill success rates across episodes to identify
    underperforming skills U^(r).

    A skill is flagged when success_rate < η.
    """

    def __init__(self, eta: float = 0.5, window: int = 20):
        """
        Parameters
        ----------
        eta    : success-rate threshold below which a skill is flagged
        window : rolling window size for computing success rate
        """
        self.eta    = eta
        self.window = window
        # skill_name -> deque of bool outcomes
        self._records: dict[str, list[bool]] = {}

    def record(self, skill_name: str, success: bool) -> None:
        if skill_name not in self._records:
            self._records[skill_name] = []
        buf = self._records[skill_name]
        buf.append(success)
        if len(buf) > self.window:
            buf.pop(0)

    def record_episode(self, episode: EpisodeLog) -> None:
        """Update all skills that appeared in an episode."""
        skills_in_episode = {s.skill_selected for s in episode.steps}
        for skill_name in skills_in_episode:
            self.record(skill_name, episode.success)

    def success_rate(self, skill_name: str) -> float:
        buf = self._records.get(skill_name, [])
        if not buf:
            return 1.0   # no data → not flagged
        return sum(buf) / len(buf)

    def underperforming_skills(self) -> list[str]:
        """Return skill names with success_rate < η."""
        return [
            name for name, buf in self._records.items()
            if len(buf) >= 5 and (sum(buf) / len(buf)) < self.eta
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Main evolution engine (Algorithm 1, evolution phase)
# ─────────────────────────────────────────────────────────────────────────────

class SkillEvolutionEngine:
    """
    Emotion-Driven Skill Evolution.

    Parameters
    ----------
    llm_callable : callable(system: str, user: str) -> str
    tracker      : SkillPerformanceTracker
    eta          : success threshold (default 0.5)
    """

    def __init__(
        self,
        llm_callable,
        tracker: Optional[SkillPerformanceTracker] = None,
        eta: float = 0.5,
    ):
        self._llm    = llm_callable
        self.tracker = tracker or SkillPerformanceTracker(eta=eta)
        self.eta     = eta

    def run_round(
        self,
        library:  SkillLibrary,
        episodes: list[EpisodeLog],
        verbose:  bool = False,
    ) -> dict[str, str]:
        """
        Execute one evolution round over a batch of episodes.

        Steps
        -----
        1. Update performance tracker with episode outcomes.
        2. Identify underperforming skills U^(r).
        3. For each s_i ∈ U^(r), aggregate diagnostics across failed episodes.
        4. Call LLM rewriter → D_i'.
        5. Update library in-place.

        Parameters
        ----------
        library  : S^(r) — modified in-place to S^(r+1)
        episodes : batch of completed EpisodeLogs
        verbose  : print progress

        Returns
        -------
        rewrites : dict mapping skill_name -> new SOP text
        """
        # Step 1: update tracker
        for ep in episodes:
            self.tracker.record_episode(ep)

        # Step 2: identify underperforming skills
        under = self.tracker.underperforming_skills()
        if not under:
            if verbose:
                print("[Evolution] No underperforming skills this round.")
            return {}

        if verbose:
            print(f"[Evolution] Underperforming skills: {under}")

        rewrites: dict[str, str] = {}

        for skill_name in under:
            try:
                skill = library[skill_name]
            except KeyError:
                continue

            # Step 3: collect diagnostics from failed episodes
            all_windows: list[DiagnosticWindow] = []
            last_outcome = 0.0
            last_success = False

            for ep in episodes:
                if ep.success:
                    continue   # only analyse failures
                deltas      = compute_shift_magnitudes(ep)
                transitions = find_transition_points(ep, deltas)
                windows     = extract_diagnostic_windows(
                    ep, transitions, skill_name, deltas
                )
                all_windows.extend(windows)
                last_outcome = ep.outcome
                last_success = ep.success

            # Step 4: build prompt and call rewriter
            prompt   = build_evolution_prompt(
                skill, all_windows, last_outcome, last_success
            )
            new_sop  = self._llm(EVOLUTION_SYSTEM, prompt).strip()

            if not new_sop:
                if verbose:
                    print(f"  [Evolution] Empty rewrite for {skill_name}, skipping.")
                continue

            # Step 5: update library (Eq. 17)
            new_skill = Skill(
                name        = skill.name,
                sop         = new_sop,
                description = skill.description,
            )
            library.replace(skill_name, new_skill)
            rewrites[skill_name] = new_sop

            if verbose:
                print(f"  [Evolution] Rewrote '{skill_name}'.")

        return rewrites

    def run_rounds(
        self,
        library:          SkillLibrary,
        episode_batches:  list[list[EpisodeLog]],
        verbose:          bool = False,
    ) -> list[dict[str, str]]:
        """
        Run R evolution rounds, one per episode batch.

        Returns list of rewrites per round.
        """
        all_rewrites = []
        for r, batch in enumerate(episode_batches):
            if verbose:
                print(f"\n=== Evolution Round {r + 1} / {len(episode_batches)} ===")
            rewrites = self.run_round(library, batch, verbose=verbose)
            all_rewrites.append(rewrites)
        return all_rewrites
