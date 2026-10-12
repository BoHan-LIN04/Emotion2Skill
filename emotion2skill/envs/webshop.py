"""
WebShop Environment Interface
==============================
Wraps the WebShop gym environment for use with Emotion2Skill.

WebShop (Yao et al., 2022): https://github.com/princeton-nlp/WebShop
Install: pip install webshop  (or clone + pip install -e .)

This module provides:
  - WebShopEnv: a thin wrapper around the gym environment
  - Observation formatting for the agent prompt
  - Metric computation: Score and Success%

The standard WebShop test split contains 500 tasks.
We report Score (0–1 continuous) and Succ.% (score >= 1.0).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class WebShopStep:
    observation: str
    reward:      float
    done:        bool
    info:        dict


class WebShopEnv:
    """
    Wrapper around the WebShop gym environment.

    Parameters
    ----------
    split       : "test" | "train" | "dev"
    max_steps   : maximum steps per episode (paper uses 15)
    observation_mode : "html" | "text" (text is cleaner for LLM agents)
    task_indices : optional list of task indices to evaluate on (subset)
    """

    def __init__(
        self,
        split:            str = "test",
        max_steps:        int = 15,
        observation_mode: str = "text",
        task_indices:     Optional[list[int]] = None,
    ):
        self.split            = split
        self.max_steps        = max_steps
        self.observation_mode = observation_mode
        self.task_indices     = task_indices
        self._env             = None
        self._step_count      = 0
        self._current_task_id = None

        self._init_env()

    def _init_env(self) -> None:
        """Lazy import so the file can be loaded without WebShop installed."""
        try:
            import gym  # noqa: F401
            # WebShop registers "WebShop-v0" on import
            import web_agent_site.envs  # noqa: F401
            import gym as _gym
            self._env = _gym.make(
                "WebShop-v0",
                observation_mode=self.observation_mode,
            )
        except ImportError:
            print(
                "[WebShopEnv] WebShop not installed. "
                "Clone https://github.com/princeton-nlp/WebShop and "
                "run `pip install -e .` in its root directory."
            )
            self._env = None

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, task_idx: Optional[int] = None) -> str:
        """
        Reset the environment to a new task.

        Parameters
        ----------
        task_idx : index into the test split (if None, cycles sequentially)

        Returns
        -------
        observation : str — the initial task instruction + product page
        """
        if self._env is None:
            return self._mock_reset(task_idx)

        self._step_count = 0
        if task_idx is not None:
            obs = self._env.reset(idx=task_idx)
        else:
            obs = self._env.reset()
        self._current_task_id = task_idx
        return self._format_observation(obs)

    def step(self, action: str) -> WebShopStep:
        """
        Execute an action string and return the next observation.

        Valid action formats (WebShop):
          - "search[<query>]"
          - "click[<button_text>]"
          - "stop[<product_url>]"  (submit purchase)

        Returns
        -------
        WebShopStep with observation, reward ∈ [0, 1], done flag, and info.
        """
        if self._env is None:
            return self._mock_step(action)

        self._step_count += 1
        obs, reward, done, info = self._env.step(action)

        if self._step_count >= self.max_steps:
            done = True

        return WebShopStep(
            observation = self._format_observation(obs),
            reward      = float(reward),
            done        = done,
            info        = info or {},
        )

    def task_instruction(self) -> str:
        """Return the current task goal as a clean string."""
        if self._env is None:
            return "Buy a blue cotton shirt under $50."
        # The goal is embedded in the first observation; extract it
        obs = self._env.observation
        if "Instruction:" in obs:
            start = obs.index("Instruction:") + len("Instruction:")
            end   = obs.find("\n", start)
            return obs[start:end].strip()
        return obs[:200]

    # ── formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_observation(raw_obs: str) -> str:
        """Clean up raw WebShop HTML/text observation for LLM consumption."""
        # Remove excess whitespace
        lines = [l.strip() for l in raw_obs.split("\n") if l.strip()]
        return "\n".join(lines)

    def format_context(
        self,
        instruction:  str,
        history:      list[tuple[str, str]],   # [(action, observation), ...]
        current_obs:  str,
    ) -> str:
        """
        Build the x_t context string for the skill-selection prompt.
        """
        parts = [f"Task Goal: {instruction}", ""]
        if history:
            parts.append("Previous Steps:")
            for i, (act, obs) in enumerate(history[-5:], 1):  # last 5 steps
                parts.append(f"  Step {i}: action={act}")
                parts.append(f"           obs={obs[:120]}")
            parts.append("")
        parts.append(f"Current Observation:\n{current_obs}")
        return "\n".join(parts)

    # ── metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(scores: list[float]) -> dict[str, float]:
        """
        Compute WebShop evaluation metrics.

        Parameters
        ----------
        scores : list of per-episode scores ∈ [0, 1]

        Returns
        -------
        {"score": float, "success_rate": float}
        """
        n = len(scores)
        if n == 0:
            return {"score": 0.0, "success_rate": 0.0}
        avg_score = sum(scores) / n
        success   = sum(1 for s in scores if s >= 1.0) / n * 100
        return {
            "score":        round(avg_score * 100, 1),
            "success_rate": round(success, 1),
        }

    # ── mock methods (for unit testing without WebShop installed) ─────────────

    def _mock_reset(self, task_idx: Optional[int] = None) -> str:
        self._step_count = 0
        return (
            "Instruction: Buy a blue cotton shirt with short sleeves "
            "and a price lower than $50.00.\n"
            "Search bar available."
        )

    def _mock_step(self, action: str) -> WebShopStep:
        self._step_count += 1
        done = self._step_count >= self.max_steps or action.startswith("stop[")
        reward = 0.85 if done else 0.0
        return WebShopStep(
            observation = f"[Mock] After action: {action}",
            reward      = reward,
            done        = done,
            info        = {},
        )
