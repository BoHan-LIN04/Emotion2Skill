"""
ALFWorld Environment Interface
================================
Wraps the ALFWorld text environment for use with Emotion2Skill.

ALFWorld (Shridhar et al., 2021): https://github.com/alfworld/alfworld
Install: pip install alfworld

This module provides:
  - ALFWorldEnv: wrapper around the alfworld.agents.environment
  - Task-type detection (Pick, Look, Clean, Heat, Cool, Pick2)
  - Observation formatting for the agent prompt
  - Metric computation: per-type and average Success%

Evaluation follows the standard json_2.1.1 split (134 test episodes)
across six task types.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# ── ALFWorld task type labels ─────────────────────────────────────────────────
TASK_TYPES = ["pick", "look", "clean", "heat", "cool", "pick2"]

TASK_TYPE_KEYWORDS = {
    "look":  ["examine", "look at", "look under"],
    "clean": ["clean", "rinse", "wash"],
    "heat":  ["heat", "warm", "microwave"],
    "cool":  ["cool", "chill", "fridge", "refrigerator"],
    "pick2": ["two", "both", "a couple"],
    "pick":  [],   # fallback
}


@dataclass
class ALFWorldStep:
    observation: str
    reward:      float
    done:        bool
    info:        dict


def detect_task_type(task_desc: str) -> str:
    """
    Infer the ALFWorld task type from the task description string.
    Returns one of the six task-type labels.
    """
    desc_lower = task_desc.lower()
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return task_type
    return "pick"   # default


class ALFWorldEnv:
    """
    Wrapper around the ALFWorld text environment.

    Parameters
    ----------
    data_path      : path to ALFWorld data root (contains json_2.1.1/)
    split          : "eval_out_of_distribution" (standard test split)
    max_steps      : maximum steps per episode (paper uses 50)
    batch_size     : number of parallel environments (1 for sequential eval)
    """

    def __init__(
        self,
        data_path:  str = os.environ.get("ALFWORLD_DATA", "data/alfworld"),
        split:      str = "eval_out_of_distribution",
        max_steps:  int = 50,
        batch_size: int = 1,
    ):
        self.data_path  = data_path
        self.split      = split
        self.max_steps  = max_steps
        self.batch_size = batch_size
        self._env       = None
        self._step_count = 0
        self._task_type: Optional[str] = None
        self._current_obs: str = ""

        self._init_env()

    def _init_env(self) -> None:
        try:
            import alfworld.agents.environment as aenv
            import yaml

            config_path = os.path.join(self.data_path, "base_config.yaml")
            if not os.path.exists(config_path):
                # Generate a minimal config
                config = self._default_config()
            else:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

            config["dataset"]["data_path"] = self.data_path
            config["env"]["max_steps"]     = self.max_steps

            self._env = aenv.AlfredTWEnv(config, train_eval=self.split)
            self._env = self._env.init_env(batch_size=self.batch_size)
        except ImportError:
            print(
                "[ALFWorldEnv] ALFWorld not installed. "
                "Run: pip install alfworld && alfworld-download"
            )
            self._env = None

    @staticmethod
    def _default_config() -> dict:
        return {
            "dataset": {
                "data_path": "data/alfworld",
                "eval_out_of_distribution": "data/alfworld/json_2.1.1/valid_unseen",
            },
            "env": {
                "max_steps": 50,
                "domain_randomization": False,
            },
        }

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> tuple[str, str]:
        """
        Reset to the next episode.

        Returns
        -------
        (observation, task_description) — both as strings
        """
        if self._env is None:
            return self._mock_reset()

        self._step_count = 0
        obs, info = self._env.reset()

        task_desc = info.get("extra.gamefile", [""])[0] if isinstance(info, dict) else ""
        if not task_desc:
            # fallback: extract from observation
            task_desc = obs[0] if isinstance(obs, list) else obs

        raw_obs = obs[0] if isinstance(obs, list) else obs
        self._current_obs = self._format_observation(raw_obs)
        self._task_type   = detect_task_type(task_desc)

        # Extract clean task goal sentence
        task_goal = self._extract_goal(raw_obs)
        return self._current_obs, task_goal

    def step(self, action: str) -> ALFWorldStep:
        """
        Execute a text action and return the next step.

        Valid actions are free-form text commands, e.g.:
          "go to sidetable 1"
          "pick up tomato 1"
          "heat tomato 1 with microwave 1"

        Returns
        -------
        ALFWorldStep
        """
        if self._env is None:
            return self._mock_step(action)

        self._step_count += 1
        obs, scores, dones, infos = self._env.step([action])

        raw_obs    = obs[0]   if isinstance(obs, list)    else obs
        reward     = scores[0] if isinstance(scores, list) else scores
        done       = dones[0]  if isinstance(dones, list)  else dones

        if self._step_count >= self.max_steps:
            done = True

        self._current_obs = self._format_observation(raw_obs)
        return ALFWorldStep(
            observation = self._current_obs,
            reward      = float(reward),
            done        = done,
            info        = infos[0] if isinstance(infos, list) else (infos or {}),
        )

    @property
    def task_type(self) -> Optional[str]:
        return self._task_type

    # ── formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_observation(raw: str) -> str:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return "\n".join(lines)

    @staticmethod
    def _extract_goal(obs: str) -> str:
        """Extract the task goal sentence from the initial observation."""
        for line in obs.split("\n"):
            line = line.strip()
            if line.lower().startswith("your task is"):
                return line
            if line.lower().startswith("you are"):
                return line
        # fallback
        return obs.split("\n")[0].strip()

    def format_context(
        self,
        task_goal:    str,
        history:      list[tuple[str, str]],
        current_obs:  str,
    ) -> str:
        """Build x_t context string for the skill-selection prompt."""
        parts = [f"Task: {task_goal}", ""]
        if history:
            parts.append("Previous Steps:")
            for i, (act, obs) in enumerate(history[-6:], 1):
                parts.append(f"  {i}. > {act}")
                parts.append(f"      {obs[:120]}")
            parts.append("")
        parts.append(f"Current Observation:\n{current_obs}")
        return "\n".join(parts)

    # ── metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        results: list[dict],
    ) -> dict[str, float]:
        """
        Compute per-type and average ALFWorld Success%.

        Parameters
        ----------
        results : list of {"task_type": str, "success": bool}

        Returns
        -------
        dict with keys: "pick", "look", "clean", "heat", "cool",
                        "pick2", "avg"
        """
        from collections import defaultdict
        counts   = defaultdict(int)
        successes = defaultdict(int)
        for r in results:
            tt = r["task_type"]
            counts[tt]    += 1
            successes[tt] += int(r["success"])

        metrics: dict[str, float] = {}
        for tt in TASK_TYPES:
            n   = counts.get(tt, 0)
            s   = successes.get(tt, 0)
            metrics[tt] = round(s / n * 100, 1) if n > 0 else 0.0

        total    = sum(counts.values())
        total_ok = sum(successes.values())
        metrics["avg"] = round(total_ok / total * 100, 1) if total > 0 else 0.0
        return metrics

    # ── mock methods ─────────────────────────────────────────────────────────

    def _mock_reset(self) -> tuple[str, str]:
        self._step_count = 0
        self._task_type  = "heat"
        obs  = "You are in a kitchen. A microwave is on the counter."
        goal = "Your task is to heat a tomato and put it on the counter."
        return obs, goal

    def _mock_step(self, action: str) -> ALFWorldStep:
        self._step_count += 1
        done   = self._step_count >= self.max_steps
        reward = 1.0 if done else 0.0
        return ALFWorldStep(
            observation = f"[Mock] After: {action}",
            reward      = reward,
            done        = done,
            info        = {},
        )
