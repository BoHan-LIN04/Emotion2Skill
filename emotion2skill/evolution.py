"""Emotion transition diagnostics and SOP rewriting."""
from __future__ import annotations

import dataclasses
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from .selection import Skill, SkillLibrary


def _serializable(value):
    if dataclasses.is_dataclass(value):
        return {item.name: _serializable(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Run evidence contains an unsupported value: {type(value).__name__}")


@dataclass
class StepRecord:
    step: int
    observation: str
    skill_selected: str
    template_idx: int
    confidence: float
    emotion_state: Any
    action_taken: str
    reward: float
    context: str = ""
    template_text: str = ""
    emotion_used: bool = False
    extraction_prompt: str = ""
    selection_prompt: str = ""
    selection_raw_output: str = ""
    candidate_scores: dict[str, float] = field(default_factory=dict)
    action_prompt: str = ""
    action_raw_output: str = ""
    next_observation: str = ""
    environment_info: dict = field(default_factory=dict)
    done: bool = False
    decision_seconds: float = 0.0


@dataclass
class EpisodeLog:
    episode_id: str
    steps: list[StepRecord] = field(default_factory=list)
    outcome: float = 0.0
    success: bool = False
    benchmark: str = ""
    task_id: str = ""
    task_type: str = ""
    backend: str = "unknown"
    split: str = "unknown"
    task_goal: str = ""
    run_id: str = ""
    seed: int = 42
    phase: str = "evaluation"
    round_id: int = 0
    terminated: bool = False
    truncated: bool = False
    attempts: list[dict] = field(default_factory=list)
    reflection_prompts: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    final_info: dict = field(default_factory=dict)

    def add_step(self, record: StepRecord) -> None:
        self.steps.append(record)

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        result = _serializable(self)
        result["n_steps"] = len(self.steps)
        result["synthetic"] = self.backend.casefold() in {"mock", "synthetic"}

        for step in result["steps"]:
            step["skill"] = step["skill_selected"]
            step["action"] = step["action_taken"]
        return result


def compute_shift_magnitudes(episode: EpisodeLog) -> list[float]:
    states = [_serializable(step.emotion_state) for step in episode.steps]
    if len(states) < 2 or any(not state for state in states):
        return []
    deltas = []
    for previous, current in zip(states, states[1:]):
        if len(previous) != len(current):
            raise ValueError("Emotion vectors must have a consistent dimensionality.")
        norm_previous = math.sqrt(sum(float(x) ** 2 for x in previous))
        norm_current = math.sqrt(sum(float(x) ** 2 for x in current))
        if norm_previous == 0 or norm_current == 0:
            raise ValueError("Zero emotion vectors are invalid diagnostic evidence.")
        cosine = sum(float(a) * float(b) for a, b in zip(previous, current)) / (norm_previous * norm_current)
        deltas.append(1.0 - max(-1.0, min(1.0, cosine)))
    return deltas


def find_transition_points(episode: EpisodeLog, deltas: list[float]) -> list[int]:
    """Return zero-based steps; delta[i] corresponds to step i + 1."""
    if len(deltas) < 2:
        return []
    threshold = statistics.mean(deltas) + statistics.pstdev(deltas)
    return [index + 1 for index, delta in enumerate(deltas) if delta > threshold]


@dataclass
class DiagnosticWindow:
    center_step: int
    observations: list[str]
    skills: list[str]
    templates: list[str]
    shift_mags: list[float]
    skill_of_interest: str
    contexts: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    episode_id: str = ""
    task_id: str = ""
    outcome: float = 0.0
    success: bool = False
    step_indices: list[int] = field(default_factory=list)

    def verbalize(self) -> str:
        lines = [f"Episode: {self.episode_id}; task: {self.task_id}; outcome: {self.outcome}; success: {self.success}",
                 f"Transition at zero-based step {self.center_step}"]
        for index, label in enumerate(("t-1", "t", "t+1")):
            context = self.contexts[index] if self.contexts else self.observations[index]
            action = self.actions[index] if self.actions else "(not recorded)"
            lines.append(f"Step {label}:\nContext:\n{context}\nSkill: {self.skills[index]}\n"
                         f"Emotion: {self.templates[index]}\nShift: {self.shift_mags[index]:.6f}\nAction: {action}")
        return "\n\n".join(lines)


def extract_diagnostic_windows(episode: EpisodeLog, transitions: list[int], skill_name: str,
                               deltas: list[float]) -> list[DiagnosticWindow]:
    windows, count = [], len(episode.steps)
    for center in transitions:
        if not 0 <= center < count or episode.steps[center].skill_selected != skill_name:
            continue
        indices = [max(0, center - 1), center, min(count - 1, center + 1)]
        steps = [episode.steps[index] for index in indices]
        templates = [step.template_text or (f"unencoded template {step.template_idx}" if step.template_idx >= 0
                                           else "(emotion vector collected without template encoding)") for step in steps]
        windows.append(DiagnosticWindow(center, [step.observation for step in steps],
            [step.skill_selected for step in steps], templates,
            [deltas[index - 1] if 0 < index <= len(deltas) else 0.0 for index in indices], skill_name,
            contexts=[step.context or step.observation for step in steps],
            actions=[step.action_taken for step in steps], episode_id=episode.episode_id,
            task_id=episode.task_id, outcome=episode.outcome, success=episode.success, step_indices=indices))
    return windows


VERBALIZATION_SYSTEM = "You analyze recorded agent trajectories. Use only the supplied evidence; do not invent failure causes."
VERBALIZATION_USER_TEMPLATE = """Skill: {skill_name}
Current SOP:
{current_sop}

Recorded transition contexts (each includes its own episode outcome):
{diagnostics}

Provide a 2–3 sentence summary of (1) the observed internal state shifts,
(2) any supported reason the SOP caused or failed to prevent the difficulty,
and (3) the SOP aspect to revise. State when evidence is insufficient.
A large cosine shift can have any valence; do not assume it means frustration.
Do not rewrite the SOP here."""
EVOLUTION_SYSTEM = "You improve agent skill SOPs from recorded evidence. Output only the revised SOP."
EVOLUTION_USER_TEMPLATE = """Skill: {skill_name}
Success rate: {success_rate} (underperformance threshold: {eta})

Current SOP:
{current_sop}

Episode outcomes:
{outcomes}

Critical transition diagnostics:
{diagnostics}

Transition pattern summary:
{summary}

Rewrite the SOP using supported failure patterns. Add clear preconditions,
recovery, exit and escalation criteria where the evidence supports them.
Preserve useful instructions and stay within 200 words. Revised SOP:"""


def build_evolution_prompt(skill: Skill, windows: list[DiagnosticWindow], outcome: float,
                           success: bool, delta_summary: Optional[str] = None,
                           success_rate: Optional[float] = None, eta: float = 0.4,
                           episode_outcomes: Optional[list[dict]] = None,
                           diagnostics_override: Optional[str] = None) -> str:
    diagnostics = diagnostics_override if diagnostics_override is not None else (
        "\n\n".join(window.verbalize() for window in windows) or "No skill-specific transitions were detected.")
    summary = delta_summary if delta_summary is not None else "No LLM transition summary was supplied."
    outcomes = episode_outcomes if episode_outcomes is not None else [{"outcome": outcome, "success": success}]
    return EVOLUTION_USER_TEMPLATE.format(skill_name=skill.name, current_sop=skill.sop,
        success_rate="unavailable" if success_rate is None else f"{success_rate:.6f}", eta=f"{eta:.6f}",
        outcomes=str(outcomes), diagnostics=diagnostics, summary=summary)


class SkillPerformanceTracker:
    def __init__(self, eta: float = 0.4, window: int = 100, min_episodes: int = 5):
        self.eta, self.window, self.min_episodes = eta, window, min_episodes
        self._records: dict[str, list[bool]] = {}

    def record(self, skill_name: str, success: bool) -> None:
        buffer = self._records.setdefault(skill_name, [])
        buffer.append(success)
        if len(buffer) > self.window:
            buffer.pop(0)

    def record_episode(self, episode: EpisodeLog) -> None:
        for skill in {step.skill_selected for step in episode.steps}:
            self.record(skill, episode.success)

    def success_rate(self, skill_name: str) -> float:
        values = self._records.get(skill_name, [])
        return sum(values) / len(values) if values else 1.0

    def underperforming_skills(self) -> list[str]:
        return [name for name, values in self._records.items()
                if len(values) >= self.min_episodes and self.success_rate(name) < self.eta]


class DiagnosticBudget:
    """Reduce diagnostics within tokenizer budgets; retain every source span and reduction lineage."""
    def __init__(self, llm, records):
        self.llm, self.records = llm, records
        limit = getattr(llm, "max_context_tokens", None)
        self.budget = limit - getattr(llm, "max_new_tokens", 512) if limit else None
        if self.budget is not None and self.budget <= 0:
            raise ValueError("Context budget must exceed the reserved generation budget.")

    def count(self, system, user):
        counter = getattr(self.llm, "count_prompt_tokens", None)
        if callable(counter):
            return counter(system, user)
        if self.budget is not None:
            tokenizer = getattr(self.llm, "tokenizer", None)
            prefix = getattr(self.llm, "chat_prefix", None)
            if tokenizer is None or not callable(prefix):
                raise TypeError("A bounded LLM must expose its actual tokenizer/chat prefix for prompt budgeting.")
            return len(tokenizer.encode(prefix(system, user), add_special_tokens=False))
        return None

    def fits(self, system, user):
        return self.budget is None or self.count(system, user) <= self.budget

    def call(self, system, prompt, stage, pieces):
        if not self.fits(system, prompt):
            raise ValueError(f"{stage} exceeds the tokenizer-measured input budget.")
        record = {"stage": stage, "prompt": prompt, "input_tokens": self.count(system, prompt),
                  "prompt_budget": self.budget, "source_window_ids": sorted({source for piece in pieces
                    for source in piece["sources"]}), "input_pieces": pieces}
        self.records.append(record)
        output = self.llm(system, prompt).strip()
        record["output"] = output
        if not output:
            raise ValueError("The budgeted diagnostic summarizer returned empty output.")
        return {"id": f"summary_{len(self.records):04d}", "text": output,
                "sources": record["source_window_ids"]}

    @staticmethod
    def render(pieces):
        return "\n\n".join(f"[{piece['id']}]\n{piece['text']}" for piece in pieces)

    def summarize(self, windows, prompt_fn):
        pieces = [{"id": f"window_{index:04d}", "text": window.verbalize(),
                   "sources": [f"window_{index:04d}"]} for index, window in enumerate(windows)]
        if not pieces:
            pieces = [{"id": "no_transitions", "text": "No abrupt skill-specific transitions were observed.",
                       "sources": []}]
        prompt = prompt_fn(self.render(pieces))
        if self.fits(VERBALIZATION_SYSTEM, prompt):
            return self.call(VERBALIZATION_SYSTEM, prompt, "all_windows", pieces)
        # Split oversized windows into reversible ranges using actual chat-token counts.

        fragments = []
        for piece in pieces:
            if self.fits(VERBALIZATION_SYSTEM, prompt_fn(self.render([piece]))):
                fragments.append(piece)
                continue
            cursor, ordinal = 0, 0
            while cursor < len(piece["text"]):
                identifier = f"{piece['id']}_fragment_{ordinal:04d}"
                remaining = len(piece["text"]) - cursor
                low, high, fitted = 1, remaining, 0
                while low <= high:
                    middle = (low + high) // 2
                    candidate = {"id": identifier, "text": piece["text"][cursor:cursor + middle],
                                 "sources": piece["sources"], "character_range": [cursor, cursor + middle]}
                    if self.fits(VERBALIZATION_SYSTEM, prompt_fn(self.render([candidate]))):
                        fitted, low = middle, middle + 1
                    else:
                        high = middle - 1
                if fitted == 0:
                    raise ValueError("The SOP and verbalization instructions leave no room for a diagnostic fragment.")
                fragment = {"id": identifier, "text": piece["text"][cursor:cursor + fitted],
                            "sources": piece["sources"], "character_range": [cursor, cursor + fitted]}


                fragments.append(self.call(VERBALIZATION_SYSTEM, prompt_fn(self.render([fragment])),
                                           "window_fragment", [fragment]))
                cursor += fitted
                ordinal += 1
        pieces = fragments
        for depth in range(16):
            combined = prompt_fn(self.render(pieces))
            if self.fits(VERBALIZATION_SYSTEM, combined):
                return self.call(VERBALIZATION_SYSTEM, combined, "merged_diagnosis", pieces)
            batches, batch = [], []
            for piece in pieces:
                candidate = batch + [piece]
                if self.fits(VERBALIZATION_SYSTEM, prompt_fn(self.render(candidate))):
                    batch = candidate
                else:
                    if not batch:
                        raise ValueError("A generated diagnostic summary exceeds the reserved merge budget.")
                    batches.append(batch)
                    batch = [piece]
                    if not self.fits(VERBALIZATION_SYSTEM, prompt_fn(self.render(batch))):
                        raise ValueError("A generated diagnostic summary exceeds the reserved merge budget.")
            if batch:
                batches.append(batch)
            updated = [self.call(VERBALIZATION_SYSTEM, prompt_fn(self.render(batch)),
                                 f"diagnostic_batch_depth_{depth}", batch) for batch in batches]
            if len(updated) >= len(pieces) and sum(len(item["text"]) for item in updated) >= sum(
                    len(item["text"]) for item in pieces):
                raise ValueError("Diagnostic summaries did not shrink; increase context or reduce the generation budget.")
            pieces = updated
        raise ValueError("Diagnostic reduction exceeded 16 levels; all attempted evidence is preserved.")


class SkillEvolutionEngine:
    def __init__(self, llm_callable, tracker: Optional[SkillPerformanceTracker] = None,
                 eta: float = 0.4, diagnostics_enabled: bool = True):
        self._llm = llm_callable
        self.tracker = tracker if tracker is not None else SkillPerformanceTracker(eta=eta)
        self.eta, self.diagnostics_enabled = eta, diagnostics_enabled
        self.last_evidence: dict = {}

    def run_round(self, library: SkillLibrary, episodes: list[EpisodeLog], verbose: bool = False) -> dict[str, str]:
        for episode in episodes:
            self.tracker.record_episode(episode)
        rewrites = {}
        self.last_evidence = {"episode_ids": [episode.episode_id for episode in episodes],
            "diagnostics_enabled": self.diagnostics_enabled, "eta": self.eta,
            "min_episodes": self.tracker.min_episodes, "performance_window": self.tracker.window,
            "outcome": statistics.mean([episode.outcome for episode in episodes]) if episodes else 0.0,
            "success_rate": statistics.mean([episode.success for episode in episodes]) if episodes else 0.0,
            "skills": {}, "trajectories": []}
        transitions_by_episode = {}
        if self.diagnostics_enabled:
            for episode in episodes:
                deltas = compute_shift_magnitudes(episode)
                points = find_transition_points(episode, deltas)
                transitions_by_episode[episode.episode_id] = (deltas, points)
                self.last_evidence["trajectories"].append({"episode_id": episode.episode_id,
                    "task_id": episode.task_id, "deltas": deltas, "transition_points": points})
        for skill_name in self.tracker.underperforming_skills():
            relevant = [episode for episode in episodes
                        if any(step.skill_selected == skill_name for step in episode.steps)]
            failures = [episode for episode in relevant if not episode.success]
            if not failures or skill_name not in library.names():
                continue
            skill = library[skill_name]
            windows = []
            if self.diagnostics_enabled:
                for episode in failures:
                    deltas, points = transitions_by_episode[episode.episode_id]
                    windows.extend(extract_diagnostic_windows(episode, points, skill_name, deltas))
            entry = {"success_rate": self.tracker.success_rate(skill_name), "old_sop": skill.sop,
                     "windows": _serializable(windows), "verbalization_records": [], "accepted": False}
            self.last_evidence["skills"][skill_name] = entry
            budget = DiagnosticBudget(self._llm, entry["verbalization_records"])
            verbalization_prompt, summary = "", "Outcome-only evolution: emotion diagnostics intentionally withheld."
            if self.diagnostics_enabled:
                def verbalization_prompt_fn(diagnostics):
                    return VERBALIZATION_USER_TEMPLATE.format(skill_name=skill_name,
                        current_sop=skill.sop, diagnostics=diagnostics)
                merged = budget.summarize(windows, verbalization_prompt_fn)
                summary = merged["text"]
                verbalization_prompt = entry["verbalization_records"][-1]["prompt"]
                entry["covered_window_ids"] = merged["sources"]
                expected = {f"window_{index:04d}" for index in range(len(windows))}
                if set(merged["sources"]) != expected:
                    raise RuntimeError("Hierarchical diagnostic reduction omitted source windows.")
            outcomes = [{"episode_id": ep.episode_id, "task_id": ep.task_id,
                         "outcome": ep.outcome, "success": ep.success} for ep in relevant]
            prompt = build_evolution_prompt(skill, windows, failures[-1].outcome, False,
                delta_summary=summary, success_rate=self.tracker.success_rate(skill_name),
                eta=self.eta, episode_outcomes=outcomes)
            entry["episode_outcomes"] = outcomes
            entry["rewriter_diagnostics_mode"] = "full_windows"
            if not budget.fits(EVOLUTION_SYSTEM, prompt):
                compact_outcome = [{"task_ids": [item["task_id"] for item in outcomes],
                    "episodes": len(outcomes), "successes": sum(item["success"] for item in outcomes),
                    "mean_score": statistics.mean(item["outcome"] for item in outcomes),
                    "min_score": min(item["outcome"] for item in outcomes),
                    "max_score": max(item["outcome"] for item in outcomes)}]
                prompt = build_evolution_prompt(skill, [], failures[-1].outcome, False,
                    delta_summary=summary, success_rate=self.tracker.success_rate(skill_name),
                    eta=self.eta, episode_outcomes=compact_outcome,
                    diagnostics_override=f"All {len(windows)} complete diagnostic windows are archived. "
                        "The merged transition diagnosis below covers every source window via recorded batches/fragments.")
                entry["rewriter_diagnostics_mode"] = "hierarchical_summary_all_sources"
                entry["compact_round_outcomes"] = compact_outcome
            if not budget.fits(EVOLUTION_SYSTEM, prompt):
                raise ValueError("The current SOP, task IDs, outcomes and merged diagnosis exceed the input budget.")
            entry.update({"verbalization_prompt": verbalization_prompt, "summary": summary,
                          "rewrite_prompt": prompt, "rewrite_input_tokens": budget.count(EVOLUTION_SYSTEM, prompt),
                          "prompt_budget": budget.budget})
            raw = self._llm(EVOLUTION_SYSTEM, prompt).strip()
            rewrite_attempts = [{"prompt": prompt, "output": raw}]
            entry["rewrite_attempts"] = rewrite_attempts
            # Request one length correction while preserving procedural steps.
            if len(raw.split()) > 200:
                correction_prompt = ("Compress this SOP to at most 200 words while preserving its "
                                     "conditions and steps. Output only the SOP:\n" + raw)
                if not budget.fits(EVOLUTION_SYSTEM, correction_prompt):
                    raise ValueError("SOP correction exceeds the reserved prompt budget; previous outputs are preserved.")
                raw = self._llm(EVOLUTION_SYSTEM, correction_prompt).strip()
                rewrite_attempts.append({"prompt": correction_prompt, "output": raw})
            valid = bool(raw) and len(raw.split()) <= 200
            entry.update({"rewrite_output": raw, "rewrite_attempts": rewrite_attempts, "accepted": valid})
            if valid:
                library.replace(skill_name, Skill(skill.name, raw, skill.description))
                rewrites[skill_name] = raw
                if verbose:
                    print(f"[Evolution] Rewrote {skill_name}")
        return rewrites

    def run_rounds(self, library: SkillLibrary, episode_batches: list[list[EpisodeLog]], verbose: bool = False):
        return [self.run_round(library, batch, verbose=verbose) for batch in episode_batches]
