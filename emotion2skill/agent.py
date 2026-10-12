"""
Emotion2Skill Agent  (Algorithm 1)
====================================
Implements the full online loop from Algorithm 1 of the paper:

  FOR each episode:
    FOR each decision step t:
      1. Extract e_t from the residual stream (EmotionExtractor)
      2. Encode (r_t, c_t) via EmotionEncoder
      3. Build augmented prompt P_t
      4. Select skill s_t* via LLM (SkillSelector)
      5. Translate skill → concrete action via LLM (ActionExecutor)
      6. Step environment; log StepRecord
    END FOR
  END FOR

  FOR each evolution round r:
    run SkillEvolutionEngine.run_round(library, batch)
  END FOR

The agent wraps Qwen3 (HuggingFace) in non-thinking mode with greedy
decoding, and provides a unified LLM callable used by all sub-modules.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .encoder import EmotionEncoder, EmotionSummary
from .evolution import (
    EpisodeLog,
    SkillEvolutionEngine,
    SkillPerformanceTracker,
    StepRecord,
)
from .extraction import EmotionExtractor
from .selection import SelectionResult, SkillLibrary, SkillSelector


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3 LLM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Qwen3LLM:
    """
    Non-thinking mode, greedy decoding wrapper around Qwen3.

    Used as the single `llm_callable` threaded through all sub-modules.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        self.device         = device
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        ).eval()

    def __call__(self, system: str, user: str) -> str:
        """
        Call the LLM with a system + user prompt.
        Returns the generated text (non-thinking mode, greedy decoding).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # non-thinking mode
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # greedy
                temperature=1.0,
                repetition_penalty=1.05,
            )

        generated = ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Action executor: skill SOP → concrete environment action
# ─────────────────────────────────────────────────────────────────────────────

ACTION_SYSTEM = """\
You are an autonomous agent. Given the current observation and the selected \
skill's instructions, generate the SINGLE next action to execute in the \
environment. Output ONLY the action string — nothing else."""

ACTION_USER_TEMPLATE = """\
## Current Observation
{observation}

## Selected Skill
{skill_name}

## Skill SOP (Standard Operating Procedure)
{skill_sop}

## History (last 4 steps)
{history_str}

## Instruction
Generate the single next action to execute. \
For WebShop: use format like "search[query]" or "click[button]" or "stop[url]". \
For ALFWorld: use plain English like "pick up tomato 1" or "go to microwave 1". \
Action:\
"""


class ActionExecutor:
    """Translates a selected skill into a concrete environment action."""

    def __init__(self, llm_callable):
        self._llm = llm_callable

    def execute(
        self,
        observation:  str,
        skill_name:   str,
        skill_sop:    str,
        history:      list[tuple[str, str]],
    ) -> str:
        history_str = "\n".join(
            f"  [{i+1}] > {act}\n       {obs[:80]}"
            for i, (act, obs) in enumerate(history[-4:])
        ) if history else "(no prior steps)"

        user = ACTION_USER_TEMPLATE.format(
            observation = observation[:500],
            skill_name  = skill_name,
            skill_sop   = skill_sop,
            history_str = history_str,
        )
        return self._llm(ACTION_SYSTEM, user).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Agent configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    All hyper-parameters for the Emotion2Skill agent.
    """
    # Model
    model_name_or_path: str    = "Qwen/Qwen3-8B"
    model_key:          str    = "qwen3-8b"         # for extractor
    device:             str    = "cuda"

    # Extractor
    extractor_path:     Optional[str] = None         # precomputed W_e path
    stories_path:       str    = "data/emotion_stories/stories.json"
    goemotions_test:    str    = "data/goemotions/test.json"

    # Encoder
    encoder_path:       Optional[str] = None         # precomputed encoder ckpt
    tau:                float  = 0.3                 # confidence gate threshold

    # Evolution
    evolution_rounds:   int    = 5
    eta:                float  = 0.5                 # underperformance threshold
    evolution_window:   int    = 20                  # rolling window

    # Episode
    max_steps:          int    = 50

    # Output
    log_dir:            str    = "logs"


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────────────────────────────────────

class Emotion2SkillAgent:
    """
    Full Emotion2Skill agent implementing Algorithm 1.

    Usage
    -----
    >>> agent = Emotion2SkillAgent(config)
    >>> agent.load()                          # load extractor + encoder
    >>> results = agent.run_episodes(env, library, n_episodes=134)
    """

    def __init__(self, config: AgentConfig):
        self.config = config

        # LLM (shared across all sub-modules)
        self.llm = Qwen3LLM(
            config.model_name_or_path,
            device=config.device,
        )

        # Sub-modules (populated in load())
        self.extractor: Optional[EmotionExtractor] = None
        self.encoder:   Optional[EmotionEncoder]   = None
        self.selector   = SkillSelector(self.llm)
        self.executor   = ActionExecutor(self.llm)
        self.tracker    = SkillPerformanceTracker(
            eta=config.eta, window=config.evolution_window
        )
        self.evolver    = SkillEvolutionEngine(
            self.llm, self.tracker, eta=config.eta
        )

        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # ── setup ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load (or build) the emotion extractor and encoder."""
        cfg = self.config

        # Extractor
        if cfg.extractor_path and Path(cfg.extractor_path).exists():
            print(f"Loading extractor from {cfg.extractor_path}")
            self.extractor = EmotionExtractor.load(
                cfg.extractor_path, cfg.model_name_or_path, cfg.device
            )
        else:
            print("Building extractor (this takes a while)…")
            self.extractor = EmotionExtractor(
                cfg.model_name_or_path, cfg.model_key, cfg.device
            )
            self.extractor.build(
                stories_path        = cfg.stories_path,
                goemotions_test_path = cfg.goemotions_test,
                save_path           = cfg.extractor_path,
            )

        # Encoder
        self.encoder = EmotionEncoder(
            checkpoint_path = cfg.encoder_path,
            tau             = cfg.tau,
            device          = cfg.device,
        )

    # ── single-step logic ─────────────────────────────────────────────────────

    def _get_emotion(self, prompt_text: str) -> Optional[EmotionSummary]:
        """Extract emotion state and encode it into a summary."""
        if self.extractor is None or self.encoder is None:
            return None
        try:
            e_t    = self.extractor.get_emotion_state(prompt_text)
            return self.encoder.encode(e_t)
        except Exception as e:
            print(f"  [Emotion] extraction failed: {e}")
            return None

    def _step(
        self,
        t:            int,
        observation:  str,
        context:      str,
        library:      SkillLibrary,
        history:      list[tuple[str, str]],
        use_emotion:  bool = True,
    ) -> tuple[StepRecord, str]:
        """
        Execute one decision step.

        Returns
        -------
        (StepRecord, action_string)
        """
        # 1. Emotion extraction (on the context string used as prompt seed)
        emotion: Optional[EmotionSummary] = (
            self._get_emotion(context) if use_emotion else None
        )

        # 2. Skill selection
        sel: SelectionResult = self.selector.select(
            context=context, library=library, emotion=emotion
        )

        # 3. Action generation from selected skill
        skill    = library[sel.skill_name]
        action   = self.executor.execute(
            observation=observation,
            skill_name=skill.name,
            skill_sop=skill.sop,
            history=history,
        )

        # 4. Build step record
        record = StepRecord(
            step           = t,
            observation    = observation,
            skill_selected = sel.skill_name,
            template_idx   = emotion.template_idx if emotion else 0,
            confidence     = emotion.confidence   if emotion else 0.0,
            emotion_state  = (
                emotion.emotion_state if emotion else torch.zeros(27)
            ),
            action_taken   = action,
            reward         = 0.0,  # filled in after env.step()
        )

        return record, action

    # ── episode loop ──────────────────────────────────────────────────────────

    def run_episode(
        self,
        env,
        library:     SkillLibrary,
        episode_id:  str        = "ep_0",
        use_emotion: bool       = True,
        verbose:     bool       = False,
        format_context_fn=None,  # env-specific context formatter
    ) -> EpisodeLog:
        """
        Run a single episode.

        Parameters
        ----------
        env               : WebShopEnv or ALFWorldEnv (after reset() called)
        library           : current skill library S
        episode_id        : string identifier for logging
        use_emotion       : if False, fall back to text-only selection
        format_context_fn : callable(task_goal, history, obs) -> str
                            (bound to the env's format_context method)

        Returns
        -------
        EpisodeLog
        """
        log     = EpisodeLog(episode_id=episode_id)
        history: list[tuple[str, str]] = []

        # Get initial observation
        if hasattr(env, "task_instruction"):
            task_goal = env.task_instruction()
            observation = ""
        else:
            observation, task_goal = "", ""

        for t in range(self.config.max_steps):
            if not observation and hasattr(env, "_current_obs"):
                observation = env._current_obs

            # Build context x_t
            if format_context_fn is not None:
                context = format_context_fn(task_goal, history, observation)
            else:
                context = f"Task: {task_goal}\n\nObservation: {observation}"

            # Decision step
            record, action = self._step(
                t=t,
                observation=observation,
                context=context,
                library=library,
                history=history,
                use_emotion=use_emotion,
            )

            if verbose:
                admitted = "✓" if (record.confidence >= self.config.tau) else "✗"
                print(
                    f"  t={t:2d} | skill={record.skill_selected:20s} | "
                    f"c={record.confidence:.2f}{admitted} | "
                    f"action={action[:60]}"
                )

            # Environment step
            step_result = env.step(action)
            record.reward = step_result.reward
            log.add_step(record)
            history.append((action, step_result.observation))
            observation = step_result.observation

            if step_result.done:
                log.outcome = step_result.reward
                log.success = step_result.reward >= 1.0
                break

        if verbose:
            print(f"  Episode {episode_id}: outcome={log.outcome:.3f} "
                  f"{'SUCCESS' if log.success else 'FAILURE'}")

        return log

    # ── multi-episode evaluation ───────────────────────────────────────────────

    def run_episodes(
        self,
        env,
        library:         SkillLibrary,
        n_episodes:      int   = 134,
        use_emotion:     bool  = True,
        evolution:       bool  = True,
        episode_per_round: int = 20,
        verbose:         bool  = False,
        format_context_fn=None,
        reset_kwargs:    Optional[dict] = None,
    ) -> list[EpisodeLog]:
        """
        Run n_episodes episodes with periodic evolution rounds.

        Parameters
        ----------
        env               : initialised environment
        library           : skill library (modified in-place during evolution)
        n_episodes        : total episodes to run
        use_emotion       : enable emotion augmentation
        evolution         : enable skill evolution
        episode_per_round : batch size between evolution rounds
        verbose           : print per-step info

        Returns
        -------
        all_logs : list of EpisodeLog
        """
        all_logs: list[EpisodeLog]        = []
        current_batch: list[EpisodeLog]   = []
        round_id = 0

        for ep_idx in range(n_episodes):
            ep_id = f"ep_{ep_idx:04d}"

            # Reset environment
            kw = reset_kwargs or {}
            if hasattr(env, "task_instruction"):
                # WebShop
                obs = env.reset(task_idx=ep_idx, **kw)
                task_goal = env.task_instruction()
                fmt_fn = lambda tg, h, o: env.format_context(tg, h, o)
            else:
                # ALFWorld
                obs, task_goal = env.reset(**kw)
                fmt_fn = lambda tg, h, o: env.format_context(tg, h, o)

            if format_context_fn is not None:
                fmt_fn = format_context_fn

            log = self.run_episode(
                env            = env,
                library        = library,
                episode_id     = ep_id,
                use_emotion    = use_emotion,
                verbose        = verbose,
                format_context_fn = fmt_fn,
            )
            all_logs.append(log)
            current_batch.append(log)

            # Evolution round at the end of each batch
            if evolution and len(current_batch) >= episode_per_round:
                round_id += 1
                if verbose:
                    print(f"\n[Round {round_id}] Running skill evolution…")
                self.evolver.run_round(library, current_batch, verbose=verbose)
                self._save_library(library, round_id)
                current_batch = []

        # Final partial batch
        if evolution and current_batch:
            round_id += 1
            self.evolver.run_round(library, current_batch, verbose=verbose)
            self._save_library(library, round_id)

        self._save_logs(all_logs)
        return all_logs

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_library(self, library: SkillLibrary, round_id: int) -> None:
        path = Path(self.config.log_dir) / f"library_round_{round_id:03d}.json"
        data = [
            {"name": s.name, "description": s.description, "sop": s.sop}
            for s in library.skills
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_logs(self, logs: list[EpisodeLog]) -> None:
        path = Path(self.config.log_dir) / "episode_logs.jsonl"
        with open(path, "w") as f:
            for log in logs:
                record = {
                    "episode_id": log.episode_id,
                    "outcome":    log.outcome,
                    "success":    log.success,
                    "n_steps":    len(log.steps),
                    "steps": [
                        {
                            "t":             s.step,
                            "skill":         s.skill_selected,
                            "confidence":    round(s.confidence, 3),
                            "template_idx":  s.template_idx,
                            "action":        s.action_taken[:100],
                            "reward":        s.reward,
                        }
                        for s in log.steps
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
