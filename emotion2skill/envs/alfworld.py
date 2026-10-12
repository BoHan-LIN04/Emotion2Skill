"""ALFWorld adapter with task manifests and complete upstream config."""
from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path
from .common import BenchmarkStep, ManifestEnv
ALFWorldStep = BenchmarkStep
TASK_TYPES = ["pick", "look", "clean", "heat", "cool", "pick2"]
OFFICIAL_TYPES = {"pick_and_place_simple": "pick", "look_at_obj_in_light": "look",
    "pick_clean_then_place_in_recep": "clean", "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "pick2"}

def detect_task_type(task_desc):
    for name, short in OFFICIAL_TYPES.items():
        if name in task_desc:
            return short
    text = task_desc.lower()
    for name, words in {"look": ["examine", "look at"], "clean": ["clean", "wash"],
        "heat": ["heat", "warm"], "cool": ["cool", "chill"], "pick2": ["two", "both"]}.items():
        if any(word in text for word in words):
            return name
    return "pick"

class ALFWorldEnv(ManifestEnv):
    benchmark = "alfworld"
    action_instruction = 'Use ALFWorld commands: go to receptacle N, take object N from receptacle N, put object N in/on receptacle N, open/close receptacle N, clean/heat/cool object N with receptacle N, examine object N, use desklamp N or use floorlamp N. Look tasks require using a lamp while holding the target object.'

    def __init__(self, manifest_path=None, data_path=None, split="test", max_steps=50,
                 mock=False, env_factory=None, config_path=None, batch_size=1):
        if batch_size != 1:
            raise ValueError("The sequential adapter requires batch_size=1.")
        self._configure(manifest_path, split, mock, max_steps)
        self.data_path = str(Path(data_path or os.environ.get("ALFWORLD_DATA", "data/alfworld")).expanduser().resolve())
        self._factory, self._env, self._active_task = env_factory, None, None
        self._config = self._default_config()
        if not mock:
            if config_path:
                import yaml
                self._config = yaml.safe_load(Path(config_path).read_text())
            if env_factory is None:
                try:
                    import alfworld.agents.environment as aenv
                    self._factory = aenv.AlfredTWEnv
                except ImportError as exc:
                    raise RuntimeError("ALFWorld is unavailable. Install its pinned version and data, or explicitly use --mock for a synthetic smoke test.") from exc

    def _default_config(self):
        root = Path(self.data_path)
        return {"dataset": {"data_path": str(root / "json_2.1.1/train"),
                "eval_id_data_path": str(root / "json_2.1.1/valid_seen"),
                "eval_ood_data_path": str(root / "json_2.1.1/valid_unseen"),
                "num_train_games": -1, "num_eval_games": -1},
            "logic": {"domain": str(root / "logic/alfred.pddl"), "grammar": str(root / "logic/alfred.twl2")},
            "env": {"type": "AlfredTWEnv", "domain_randomization": False, "task_types": [1, 2, 3, 4, 5, 6],
                "expert_timeout_steps": 150, "expert_type": "handcoded", "goal_desc_human_anns_prob": 0.0},
            "general": {"training_method": "dagger"},
            "dagger": {"training": {"max_nb_steps_per_episode": self.max_steps}},
            "rl": {"training": {"max_nb_steps_per_episode": self.max_steps}}}

    def _backend_for_task(self, task):
        gamefile = Path(task["gamefile"]).expanduser()
        if not gamefile.is_absolute():
            gamefile = Path(self.data_path) / gamefile
        gamefile = gamefile.resolve()
        if not gamefile.is_file():
            raise FileNotFoundError(f"Missing manifest gamefile: {gamefile}")
        if task.get("gamefile_sha256") and hashlib.sha256(gamefile.read_bytes()).hexdigest() != task["gamefile_sha256"]:
            raise ValueError(f"ALFWorld gamefile hash mismatch: {gamefile}")
        if self.task_id != self._active_task:
            if self._env is not None and callable(getattr(self._env, "close", None)):
                self._env.close()
            config = copy.deepcopy(self._config)
            directory = str(gamefile.parent)
            config["dataset"].update(data_path=directory, eval_id_data_path=directory,
                eval_ood_data_path=directory, num_train_games=1, num_eval_games=1)
            stage = {"train": "train", "dev": "eval_in_distribution", "test": "eval_out_of_distribution"}[self.split]
            wrapper = self._factory(config, train_eval=stage)
            if hasattr(wrapper, "game_files"):
                files = [str(Path(f).resolve()) for f in wrapper.game_files]
                if files != [str(gamefile)]:
                    raise RuntimeError(f"ALFWorld loaded unexpected tasks for {self.task_id}: {files}")
            self._env = wrapper.init_env(batch_size=1)
            self._active_task = self.task_id
        return self._env

    def close(self):
        if self._env is not None and callable(getattr(self._env, "close", None)):
            self._env.close()
        self._env, self._active_task = None, None

    def reset(self, task_idx=None):
        task = self._choose_task(task_idx)
        if self.mock:
            self._current_obs = "[Synthetic] A tomato and microwave are visible."
            self._goal = "Synthetic task: heat the tomato and put it on the counter."
        else:
            self._env = self._backend_for_task(task)
            observations, info = self._env.reset()
            obs = observations[0] if isinstance(observations, (list, tuple)) else observations
            self._current_obs = str(obs)
            self._goal = next((line.strip() for line in str(obs).splitlines() if line.strip().lower().startswith("your task is")), str(obs))
            self.task_type = task.get("task_type") or detect_task_type(task["gamefile"])
            if self.task_type not in TASK_TYPES:
                raise ValueError(f"Unknown manifest task type: {self.task_type}")
        return self._current_obs, self._goal

    def step(self, action):
        self._step_count += 1
        if self.mock:
            done = self._step_count >= self.max_steps
            reward, success = (1.0 if done else 0.0), done
            obs, info = f"[Synthetic] {action}", {"synthetic": True}
        else:
            observations, rewards, dones, info = self._env.step([action])
            first = lambda value: value[0] if isinstance(value, (list, tuple)) else value
            obs, reward, done = first(observations), float(first(rewards)), bool(first(dones))
            if isinstance(info, dict):
                success = bool(first(info.get("won", [False])))
            else:
                info = first(info) or {}
                success = bool(info.get("won", False))
            done = done or self._step_count >= self.max_steps
        self._current_obs = str(obs)
        return BenchmarkStep(str(obs), float(reward), bool(done), success, info or {})

    @staticmethod
    def compute_metrics(results):
        if not results:
            raise ValueError("Cannot report metrics for an empty evaluation.")
        metrics = {}
        for task_type in TASK_TYPES:
            group = [r for r in results if r["task_type"] == task_type]
            metrics[task_type] = 100 * sum(r["success"] for r in group) / len(group) if group else None
        metrics["avg"] = 100 * sum(r["success"] for r in results) / len(results)
        return metrics
