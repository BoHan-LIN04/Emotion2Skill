"""Adapter for the official WebAgentTextEnv-v0 API."""
from __future__ import annotations
from .common import BenchmarkStep, ManifestEnv
WebShopStep = BenchmarkStep

class WebShopEnv(ManifestEnv):
    benchmark = "webshop"
    action_instruction = 'Use search[query] or click[button/ASIN/option]. Purchase with click[buy now].'

    def __init__(self, manifest_path=None, split="test", max_steps=15,
                 observation_mode="text", mock=False, env=None, task_indices=None):
        self._configure(manifest_path, split, mock, max_steps)
        if task_indices is not None:
            self.tasks = [self.tasks[i] for i in task_indices]
        self.observation_mode, self._env = observation_mode, env
        if not mock and env is None:
            try:
                import gym
                import web_agent_site.envs
                options = dict(self.manifest.get("environment", {}))
                if set(options) - {"file_path", "num_products", "human_goals", "show_attrs"}:
                    raise ValueError("Unsupported WebShop manifest environment option")
                self._env = gym.make("WebAgentTextEnv-v0", observation_mode=observation_mode, **options)
            except Exception as exc:
                raise RuntimeError("WebShop initialization failed. Install its pinned source, product data and search index; use --mock only for synthetic smoke tests.") from exc

    def reset(self, task_idx=None):
        task = self._choose_task(task_idx)
        if self.mock:
            self._goal = "Synthetic task: buy a blue cotton shirt under $50."
            self._current_obs = self._goal + "\nSearch bar available."
        else:
            value = self._env.reset(session=task["index"])
            obs = value[0] if isinstance(value, tuple) else value
            if not isinstance(obs, str):
                raise TypeError("WebShop reset must return a text observation.")
            self._current_obs = obs
            inner = getattr(self._env, "unwrapped", self._env)
            self._goal = inner.get_instruction_text()
        return self._current_obs, self._goal

    def step(self, action):
        self._step_count += 1
        if self.mock:
            done = self._step_count >= self.max_steps or action.lower() == "click[buy now]"
            reward = 0.85 if done else 0.0
            obs, info = f"[Synthetic] {action}", {"synthetic": True}
        else:
            value = self._env.step(action)
            if len(value) == 5:
                obs, reward, terminated, truncated, info = value
                done = terminated or truncated
            else:
                obs, reward, done, info = value
            done = bool(done) or self._step_count >= self.max_steps
        self._current_obs = str(obs)
        return BenchmarkStep(self._current_obs, float(reward), bool(done), bool(done and reward >= 1), info or {})

    def close(self):
        if self._env is not None and callable(getattr(self._env, "close", None)):
            self._env.close()

    @staticmethod
    def compute_metrics(scores):
        if not scores:
            raise ValueError("Cannot report metrics for an empty evaluation.")
        return {"score": sum(scores) / len(scores) * 100,
                "success_rate": sum(s >= 1 for s in scores) / len(scores) * 100}
