"""LLM routing, action execution and run logging."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
import time
import uuid
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .evolution import EpisodeLog, SkillEvolutionEngine, SkillPerformanceTracker, StepRecord
from .selection import SELECTION_SYSTEM, SkillLibrary, SkillSelector, build_selection_prompt


def validate_inference_assets(config, require_encoder: bool = True) -> dict:
    """Validate checkpoints before loading the backbone."""
    import pickle
    import numpy as np
    if not config.extractor_path or not Path(config.extractor_path).is_file():
        raise FileNotFoundError("A built extractor checkpoint is required; build it offline before running.")
    if require_encoder and config.encoder_mode == "learned" and (
            not config.encoder_path or not Path(config.encoder_path).is_file()):
        raise FileNotFoundError("Full emotion inference requires a trained encoder checkpoint.")
    with open(config.extractor_path, "rb") as stream:
        payload = pickle.load(stream)
    if payload.get("format_version") != 2 or payload.get("model_key") != config.model_key:
        raise ValueError("Extractor checkpoint schema/model does not match the requested backbone.")
    matrices = payload.get("matrices", {})
    if config.extractor_mode not in matrices or payload.get("L_star") is None:
        raise ValueError("Extractor checkpoint lacks the requested feature mode or extraction layer.")
    matrix = np.asarray(matrices[config.extractor_mode])
    from .extraction import LAYER_GRID
    expected = {"qwen3-8b": (4096, 36), "qwen3-14b": (5120, 40)}
    if config.model_key not in expected:
        raise ValueError("Only explicitly declared Qwen3-8B/14B backbone geometries are supported.")
    width, layers = expected[config.model_key]
    if matrix.shape != (27, width) or not np.isfinite(matrix).all():
        raise ValueError(f"Extractor matrix must be finite with shape {(27, width)} for {config.model_key}.")
    layer = payload["L_star"]
    if not isinstance(layer, int) or not 0 <= layer < layers or layer not in LAYER_GRID[config.model_key]:
        raise ValueError("The extractor layer is outside the declared backbone candidate grid.")
    if payload.get("metadata", {}).get("selection_split") != "validation":
        raise ValueError("Extractor checkpoint must record validation-only layer selection.")
    if config.extractor_mode == "generic_pca" and payload.get("metadata", {}).get(
            "generic_pca", {}).get("fit_split") != "train":
        raise ValueError("Generic PCA features require a recorded train-only fit.")
    metadata = payload.get("metadata", {})
    if Path(config.model_name_or_path).expanduser().is_dir():
        from .provenance import local_model_identity
        identity = local_model_identity(config.model_name_or_path)
        if metadata.get("model_identity") != identity:
            raise ValueError("Local backbone/tokenizer content does not match the extractor model identity.")
        revision = None
    else:
        revision = metadata.get("model_revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Remote extractor checkpoints require the resolved immutable model commit SHA.")
        if getattr(config, "model_revision", None) not in (None, revision):
            raise ValueError("Requested model revision differs from the extractor checkpoint revision.")
        config.model_revision = revision
    if require_encoder and config.encoder_mode == "learned":
        from .encoder import EmotionEncoder

        encoder = EmotionEncoder(config.encoder_path, tau=config.tau, device="cpu", require_trained=True,
            expected_extractor_mode=config.extractor_mode,
            expected_shuffle_context=config.shuffle_context)
        source = encoder.metadata.get("source", {})
        actual_hash = hashlib.sha256(Path(config.extractor_path).read_bytes()).hexdigest()
        if not config.mock and source.get("extractor_sha256") != actual_hash:
            raise ValueError("Encoder provenance does not match the current extractor checkpoint SHA-256.")
        declared_model = source.get("effective_config", {}).get("model", {}).get("key")
        if declared_model is not None and declared_model != config.model_key:
            raise ValueError("Encoder provenance declares a different backbone model key.")
    return {"extractor_format_version": payload["format_version"], "mode": config.extractor_mode,
            "model_revision": revision, "model_identity": metadata.get("model_identity")}


class Qwen3LLM:
    """Shared Qwen3 inference and candidate scoring."""
    def __init__(self, model_name_or_path: str, device: str = "cuda", max_new_tokens: int = 512,
                 max_context_tokens: Optional[int] = None, revision: Optional[str] = None):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.device, self.max_new_tokens = device, max_new_tokens
        self.max_context_tokens = max_context_tokens
        if revision is None and not Path(model_name_or_path).expanduser().is_dir():
            resolved = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            revision = getattr(resolved, "_commit_hash", None)
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Resolve an immutable backbone revision before loading model/tokenizer assets.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device, trust_remote_code=True, revision=revision).eval()
        self.resolved_model_revision = getattr(self.model.config, "_commit_hash", None)
        if revision and self.resolved_model_revision != revision:
            raise ValueError("Loaded model commit does not match the requested immutable revision.")
        from .provenance import local_model_identity
        self.model_identity = local_model_identity(model_name_or_path)
        self.model._emotion2skill_asset_identity = self.model_identity

    def chat_prefix(self, system: str, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def count_prompt_tokens(self, system: str, user: str) -> int:
        return len(self.tokenizer.encode(self.chat_prefix(system, user), add_special_tokens=False))

    def _inputs(self, text: str, reserve_generation: bool = False):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        reserved = self.max_new_tokens if reserve_generation else 0
        if self.max_context_tokens and inputs["input_ids"].shape[1] + reserved > self.max_context_tokens:
            raise ValueError("Prompt exceeds the configured context budget; no silent truncation is allowed.")
        return inputs.to(self.model.device)

    def __call__(self, system: str, user: str) -> str:
        import torch
        inputs = self._inputs(self.chat_prefix(system, user), reserve_generation=True)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def score_candidates(self, system: str, user: str, candidates: list[str]) -> dict[str, float]:
        """Sum identifier token log probabilities without length normalization."""
        import torch
        prefix = self._inputs(self.chat_prefix(system, user))["input_ids"]
        prefix_length = prefix.shape[1]
        scores = {}
        for candidate in candidates:
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if not ids:
                raise ValueError("A candidate identifier must contain at least one token.")
            suffix = torch.tensor([ids], dtype=prefix.dtype, device=prefix.device)
            joined = torch.cat([prefix, suffix], dim=1)
            if self.max_context_tokens and joined.shape[1] > self.max_context_tokens:
                raise ValueError("Candidate scoring exceeds the configured context budget.")
            with torch.no_grad():
                logits = self.model(input_ids=joined, attention_mask=torch.ones_like(joined)).logits
                positions = logits[0, prefix_length - 1:joined.shape[1] - 1].float()
                score = positions.log_softmax(dim=-1).gather(1, suffix[0].unsqueeze(1)).sum()
            scores[candidate] = float(score.item())
        return scores


ACTION_SYSTEM = ("You are an autonomous agent. Follow the task goal, current observation, "
                 "selected skill and environment action syntax. Output ONLY the single next action.")


class ActionExecutor:
    def __init__(self, llm_callable):
        self._llm = llm_callable
        self.last_prompt = ""
        self.last_raw_output = ""

    def execute(self, observation: str, skill_name: str, skill_sop: str,
                history: list[tuple[str, str]], context: Optional[str] = None,
                action_instruction: str = "Output the single valid environment action.",
                reasoning: bool = False) -> str:
        full_context = context if context is not None else (
            f"Current Observation:\n{observation}\nHistory:\n" +
            "\n".join(f"Action: {act}\nObservation: {obs}" for act, obs in history))
        self.last_prompt = (f"## Task and Agent Context\n{full_context}\n\n"
                            f"## Selected Skill\n{skill_name}\n\n## Skill SOP\n{skill_sop}\n\n"
                            f"## Environment Action Syntax\n{action_instruction}\n\nNext action:")
        system = ACTION_SYSTEM
        if reasoning:
            system = "Use ReAct reasoning with the selected skill, then produce a single environment action."
            self.last_prompt += "\nOutput 'Thought: <reasoning>' followed by 'Action: <next action>'."
        self.last_raw_output = self._llm(system, self.last_prompt).strip()
        if not self.last_raw_output:
            raise ValueError("The action executor returned an empty action.")
        if reasoning:
            from .baselines import parse_react_action
            return parse_react_action(self.last_raw_output)[1]
        return self.last_raw_output


@dataclass
class AgentConfig:
    model_name_or_path: str = "Qwen/Qwen3-8B"
    model_key: str = "qwen3-8b"
    model_revision: Optional[str] = None
    device: str = "cuda"
    extractor_path: Optional[str] = None
    stories_path: str = "data/emotion_stories/stories.json"
    goemotions_validation: str = "data/goemotions/validation.json"
    goemotions_test: str = "data/goemotions/test.json"
    encoder_path: Optional[str] = None
    tau: float = 0.3
    evolution_rounds: int = 3
    eta: float = 0.4
    evolution_window: int = 100
    max_steps: int = 50
    max_context_tokens: Optional[int] = None
    log_dir: str = "logs"
    run_id: Optional[str] = None
    seed: int = 42
    mock: bool = False
    ablation_mode: str = "full"
    extractor_mode: str = "contrastive"
    evolution_diagnostics: bool = True
    encoder_mode: str = "learned"
    shuffle_context: bool = False
    evolution_mode: str = "emotion"
    benchmark: str = ""
    warmup_policy: str = "react_skill"


class Emotion2SkillAgent:
    def __init__(self, config: AgentConfig, llm=None, validate_extractor: bool = True):
        self.config = config
        if llm is None and config.extractor_path and validate_extractor:
            # Pin the checkpoint revision before loading model weights.

            validate_inference_assets(config, require_encoder=False)
        self.llm = llm if llm is not None else Qwen3LLM(
            config.model_name_or_path, config.device, max_context_tokens=config.max_context_tokens,
            revision=config.model_revision)
        self.extractor = None
        self.encoder = None
        self.selector = SkillSelector(self.llm)
        self.executor = ActionExecutor(self.llm)
        self.tracker = SkillPerformanceTracker(config.eta, config.evolution_window)
        self.evolver = SkillEvolutionEngine(self.llm, self.tracker, eta=config.eta,
            diagnostics_enabled=config.evolution_diagnostics and config.evolution_mode == "emotion")
        self.run_id = config.run_id or f"run_{uuid.uuid4().hex}"
        self._segments = 0
        self._active_run_path: Optional[Path] = None
        self._active_phase = "evaluation"
        self._active_round = 0
        self._action_instruction = ""

    def load(self, load_emotion: bool = True, require_encoder: bool = True) -> None:
        if not load_emotion:
            return
        validate_inference_assets(self.config, require_encoder=require_encoder)
        from .extraction import EmotionExtractor
        from .encoder import EmotionEncoder
        cfg = self.config
        if not cfg.extractor_path or not Path(cfg.extractor_path).is_file():
            raise FileNotFoundError("A built extractor checkpoint is required; build it offline before running.")
        self.extractor = EmotionExtractor.load(cfg.extractor_path, cfg.model_name_or_path, cfg.device,
            model=self.llm.model, tokenizer=self.llm.tokenizer, revision=cfg.model_revision)
        if getattr(self.extractor, "mode", "contrastive") != cfg.extractor_mode:
            self.extractor = self.extractor.with_mode(cfg.extractor_mode)
        if require_encoder and cfg.encoder_mode == "learned":
            if not cfg.encoder_path or not Path(cfg.encoder_path).is_file():
                raise FileNotFoundError("Full emotion inference requires a trained encoder checkpoint.")
            self.encoder = EmotionEncoder(checkpoint_path=cfg.encoder_path, tau=cfg.tau,
                device=cfg.device, require_trained=True,
                expected_extractor_mode=self.extractor.mode,
                expected_shuffle_context=cfg.shuffle_context)

    def _get_emotion(self, extraction_prompt: str, encode: bool = True):
        if self.extractor is None:
            raise RuntimeError("Emotion collection requires a loaded extractor.")
        state = self.extractor.get_emotion_state(extraction_prompt)
        if encode:
            if self.config.encoder_mode == "top3":
                from .extraction import GOEMOTIONS_LABELS
                indices = sorted(range(len(state)), key=lambda i: float(state[i]), reverse=True)[:3]
                text = "Top-3 active states: " + ", ".join(GOEMOTIONS_LABELS[i] for i in indices)
                summary = SimpleNamespace(template_idx=-1, template_text=text, confidence=1.0,
                    emotion_state=state, is_admitted=self.config.tau < 1.0,
                    to_prompt_str=lambda: text)
                return state, summary
            if self.encoder is None:
                raise RuntimeError("Emotion injection requires a trained encoder; use collect_emotion for warm-up.")
            summary = self.encoder.encode(state)
            if self.config.tau >= 1.0:
                summary.is_admitted = False
            return state, summary
        return state, None

    def _step(self, t: int, observation: str, context: str, library: SkillLibrary,
              history: list[tuple[str, str]], use_emotion: bool = True,
              collect_emotion: Optional[bool] = None) -> tuple[StepRecord, str]:
        collect = use_emotion if collect_emotion is None else (collect_emotion or use_emotion)
        started = time.perf_counter()
        summary, state, extraction_prompt = None, [], ""
        if collect:
            base_prompt = build_selection_prompt(context, library, None)
            if not callable(getattr(self.llm, "chat_prefix", None)):
                raise TypeError("Emotion extraction requires an LLM with the actual chat_prefix interface.")
            extraction_prompt = self.llm.chat_prefix(SELECTION_SYSTEM, base_prompt)
            state, summary = self._get_emotion(extraction_prompt, encode=use_emotion)
        selection = self.selector.select(context=context, library=library, emotion=summary)
        skill = library[selection.skill_name]
        action_context = context
        if summary is not None and summary.is_admitted:
            action_context += "\n\nAgent Internal State:\n" + summary.to_prompt_str()
        action = self.executor.execute(observation, skill.name, skill.sop, history,
            context=action_context, action_instruction=self._action_instruction,
            reasoning=collect and not use_emotion and self.config.warmup_policy == "react_skill")
        record = StepRecord(step=t, observation=observation, context=context,
            skill_selected=selection.skill_name, template_idx=summary.template_idx if summary else -1,
            template_text=summary.template_text if summary else "",
            confidence=summary.confidence if summary else 0.0, emotion_state=state,
            emotion_used=selection.emotion_used, action_taken=action, reward=0.0,
            extraction_prompt=extraction_prompt, selection_prompt=selection.prompt,
            selection_raw_output=selection.raw_output, candidate_scores=selection.candidate_scores,
            action_prompt=self.executor.last_prompt, action_raw_output=self.executor.last_raw_output,
            decision_seconds=time.perf_counter() - started)
        return record, action

    def run_episode(self, env, library: SkillLibrary, episode_id: str = "ep_0",
                    use_emotion: bool = True, collect_emotion: Optional[bool] = None,
                    verbose: bool = False, format_context_fn=None,
                    initial_observation: Optional[str] = None, task_goal: Optional[str] = None) -> EpisodeLog:
        if initial_observation is None or task_goal is None:
            raise ValueError("Pass both values returned by env.reset as initial_observation and task_goal.")
        log = EpisodeLog(episode_id=episode_id, task_id=str(getattr(env, "task_id", episode_id)),
            benchmark=getattr(env, "benchmark", self.config.benchmark),
            task_type=str(getattr(env, "task_type", "")), backend=str(getattr(env, "backend", "unknown")),
            split=str(getattr(env, "split", "unknown")), task_goal=task_goal, run_id=self.run_id,
            seed=self.config.seed, phase=self._active_phase, round_id=self._active_round)
        history: list[tuple[str, str]] = []
        reasoning_history: list[str] = []
        observation = initial_observation
        self._action_instruction = getattr(env, "action_instruction", "Output one valid environment action.")
        for t in range(self.config.max_steps):
            formatter = format_context_fn or getattr(env, "format_context", None)
            if formatter is not None:
                context = formatter(task_goal, history, observation)
            else:
                context = (f"Task Goal: {task_goal}\n\nHistory:\n" +
                    "\n".join(f"Action: {a}\nObservation: {o}" for a, o in history) +
                    f"\n\nCurrent Observation:\n{observation}")
            if reasoning_history:
                context += "\n\nPrevious ReAct reasoning:\n" + "\n".join(reasoning_history[-6:])
            record, action = self._step(t, observation, context, library, history,
                                       use_emotion=use_emotion, collect_emotion=collect_emotion)
            result = env.step(action)
            record.reward = float(result.reward)
            record.next_observation = result.observation
            record.environment_info = dict(result.info or {})
            record.done = bool(result.done)
            log.add_step(record)
            if collect_emotion and not use_emotion and self.config.warmup_policy == "react_skill":
                reasoning_history.append(record.action_raw_output)
            history.append((action, result.observation))
            observation = result.observation
            log.outcome = float(result.reward)
            log.success = bool(result.success)
            log.final_info = dict(result.info or {})
            if verbose:
                print(f"{episode_id} step={t} skill={record.skill_selected} admitted={record.emotion_used} action={action}")
            if result.done:
                log.terminated = True
                break
        log.truncated = not log.terminated
        return log

    def run_episodes(self, env, library: SkillLibrary, n_episodes: int = 134,
                     use_emotion: bool = True, collect_emotion: Optional[bool] = None,
                     evolution: bool = False, episode_per_round: int = 50, verbose: bool = False,
                     format_context_fn=None, reset_kwargs: Optional[dict] = None,
                     task_order: Optional[list] = None, seed: Optional[int] = None,
                     run_id: Optional[str] = None, phase: str = "evaluation",
                     evolution_rounds: Optional[int] = None) -> list[EpisodeLog]:
        if evolution and phase != "evolution":
            raise ValueError("Skill updates are permitted only in phase='evolution'; test libraries must be frozen.")
        if episode_per_round < 1 or n_episodes < 0:
            raise ValueError("episode_per_round must be positive and n_episodes nonnegative.")
        rounds_limit = self.config.evolution_rounds if evolution_rounds is None else evolution_rounds
        if rounds_limit < 0:
            raise ValueError("evolution_rounds must be nonnegative.")
        if seed is not None:
            self.config.seed = seed
        if run_id is not None:
            self.run_id = run_id
        order = list(task_order) if task_order is not None else list(range(n_episodes))
        if task_order is None:
            random.Random(self.config.seed).shuffle(order)
        if len(order) != n_episodes:
            raise ValueError("task_order must contain exactly n_episodes task identifiers.")
        self._active_phase, self._active_round = phase, 0
        self._active_run_path = Path(self.config.log_dir) / self.run_id / f"{phase}_{self._segments:03d}"
        self._segments += 1
        self._active_run_path.mkdir(parents=True, exist_ok=False)
        self._save_library(library, 0)
        metadata = {"run_id": self.run_id, "seed": self.config.seed, "task_order": order,
                    "phase": phase, "config": dataclasses.asdict(self.config), "evolution_enabled": evolution,
                    "backend": getattr(env, "backend", "unknown"), "split": getattr(env, "split", "unknown"),
                    "checkpoints": self._checkpoint_hashes(), "method": getattr(self, "method_name", "Emotion2Skill")}
        if callable(getattr(env, "metadata", None)):
            metadata["environment"] = env.metadata()
        metadata["baseline"] = getattr(self, "baseline_metadata", None)
        metadata["model"] = {"name_or_path": self.config.model_name_or_path,
            "requested_revision": self.config.model_revision,
            "resolved_revision": getattr(self.llm, "resolved_model_revision", None),
            "local_identity": getattr(self.llm, "model_identity", None)}
        metadata["skill_library"] = library.metadata
        self._write_json(self._active_run_path / "run_metadata.json", metadata)
        all_logs, batch = [], []
        for index, task_idx in enumerate(order):
            kw = dict(reset_kwargs or {})
            if "task_idx" in kw:
                raise ValueError("Use task_order to specify task IDs instead of reset_kwargs.task_idx.")
            observation, goal = env.reset(task_idx=task_idx, **kw)
            log = self.run_episode(env, library, episode_id=f"ep_{index:04d}", use_emotion=use_emotion,
                collect_emotion=collect_emotion, verbose=verbose, format_context_fn=format_context_fn,
                initial_observation=observation, task_goal=goal)
            all_logs.append(log)
            batch.append(log)
            self._append_log(log)
            if evolution and len(batch) >= episode_per_round and self._active_round < rounds_limit:
                self._active_round += 1
                try:
                    self.evolver.run_round(library, batch, verbose=verbose)
                finally:
                    self._save_evolution_evidence(self._active_round)
                    self._save_library(library, self._active_round)
                batch = []
        if evolution and batch and self._active_round < rounds_limit:
            self._active_round += 1
            try:
                self.evolver.run_round(library, batch, verbose=verbose)
            finally:
                self._save_evolution_evidence(self._active_round)
                self._save_library(library, self._active_round)
        return all_logs

    def _checkpoint_hashes(self) -> dict:
        hashes = {}
        for label, path in (("extractor", self.config.extractor_path), ("encoder", self.config.encoder_path)):
            if path and Path(path).is_file():
                digest = hashlib.sha256()
                with open(path, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[label] = {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}
        return hashes

    @staticmethod
    def _write_json(path: Path, value) -> None:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)

    def _save_library(self, library: SkillLibrary, round_id: int) -> None:
        if self._active_run_path is not None:
            self._write_json(self._active_run_path / f"library_round_{round_id:03d}.json",
                             [dataclasses.asdict(skill) for skill in library.skills])

    def _save_evolution_evidence(self, round_id: int) -> None:
        self._write_json(self._active_run_path / f"evolution_round_{round_id:03d}.json", self.evolver.last_evidence)

    def _append_log(self, log: EpisodeLog) -> None:
        path = self._active_run_path / "episode_logs.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")

    def _save_logs(self, logs: list[EpisodeLog]) -> None:

        for log in logs:
            self._append_log(log)
