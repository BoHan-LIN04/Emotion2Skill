"""Explicit baseline protocols and checked adapters for upstream skill methods.

Built-ins are manuscript-protocol adaptations, not upstream paper result replicas.
Primary sources: ReAct arXiv:2210.03629; Reflexion arXiv:2303.11366;
ExpeL arXiv:2308.10144 and https://github.com/LeapLabTHU/ExpeL.
MASA/AlignEvoSkill/Trace2Skill require a source-pinned external implementation;
a lexical skill router is exposed as TextSimilarity and is never called MASA.
"""
from __future__ import annotations

import hashlib
import dataclasses
import importlib
import inspect
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from .agent import ACTION_SYSTEM, AgentConfig, Emotion2SkillAgent, Qwen3LLM
from .evolution import EpisodeLog, StepRecord

PRIMARY_SOURCES = {
    "ZeroShot": None,
    "ReAct": "https://arxiv.org/abs/2210.03629",
    "Reflexion": "https://arxiv.org/abs/2303.11366",
    "ExpeL": "https://arxiv.org/abs/2308.10144",
    "MASA": "https://arxiv.org/abs/2605.30723",
    "EvoSkill": "https://arxiv.org/abs/2506.23149v2",
    "Trace2Skill": "https://arxiv.org/abs/2603.25158",
    "TextSimilarity": None,
}


class BM25:
    """Okapi BM25 with fixed k1=1.5, b=.75 and positive Robertson IDF."""
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = .75):
        if not documents:
            raise ValueError("BM25 requires a nonempty corpus.")
        self.k1, self.b = k1, b
        self.documents = [Counter(self.tokenize(text)) for text in documents]
        self.lengths = [sum(item.values()) for item in self.documents]
        self.avg_length = sum(self.lengths) / len(self.lengths) or 1.0
        df = Counter(term for item in self.documents for term in item)
        self.idf = {term: math.log(1 + (len(documents) - count + .5) / (count + .5))
                    for term, count in df.items()}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold())

    def scores(self, query: str) -> list[float]:
        terms = set(self.tokenize(query))
        scores = []
        for document, length in zip(self.documents, self.lengths):
            score = 0.0
            for term in terms:
                tf = document[term]
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            scores.append(score)
        return scores

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        values = self.scores(query)
        return sorted(range(len(values)), key=lambda index: (-values[index], index))[:k]


def parse_react_action(output: str) -> tuple[str, str]:
    """Accept a single explicit Action field; retain Thought verbatim for evidence."""
    lines = output.splitlines()
    action_lines = [(index, line.partition(":")[2].strip()) for index, line in enumerate(lines)
                    if re.match(r"^\s*Action\s*:", line, flags=re.IGNORECASE)]
    if len(action_lines) != 1 or not action_lines[0][1]:
        raise ValueError("ReAct must output exactly one nonempty 'Action:' line.")
    index, action = action_lines[0]
    # Multiline final answers/code use a fenced block after 'Action:' instead.
    if action == "```" or action.startswith("```python"):
        tail = "\n".join(lines[index + 1:])
        if not tail.rstrip().endswith("```"):
            raise ValueError("ReAct emitted an unterminated fenced action.")
        action = action + "\n" + tail
    elif index + 1 < len(lines) and any(line.strip() for line in lines[index + 1:]):
        raise ValueError("Unexpected text after the ReAct action; output the action last.")
    thought = "\n".join(lines[:index]).strip()
    return thought, action


def _experience_text(record: dict) -> str:
    return (f"Task: {record['task_goal']}\n" + "\n".join(
        f"Observation: {step['observation']}\nAction: {step['action_taken']}\n"
        f"Result: {step.get('next_observation', '')}" for step in record["steps"]))


def build_expel_memory(llm, episodes: list[EpisodeLog], out_path, allow_mock: bool = False) -> dict:
    """Extract evolving insights from actual, disjoint training trajectories.

    The retrieval policy uses BM25/top-5 as specified by this manuscript;
    upstream ExpeL uses richer extraction/retrieval options. No outcomes or
    insights are synthesized without recorded episodes and real LLM output.
    """
    if not episodes or any(episode.split != "train" for episode in episodes):
        raise ValueError("ExpeL memory can be built only from explicitly identified training episodes.")
    synthetic = any(episode.backend in {"mock", "synthetic"} for episode in episodes)
    if synthetic and not allow_mock:
        raise ValueError("Synthetic training traces require explicit allow_mock=True.")
    if not allow_mock and len({(ep.benchmark, ep.task_id) for ep in episodes}) < 50:
        raise ValueError("The manuscript ExpeL protocol requires 50 distinct training tasks.")
    insights, extraction_records = "", []
    records = [episode.to_dict() for episode in episodes]
    # Successful/failed experiences in each block are contrasted, with existing
    # insights revised after every block rather than a retrieval-only alias.
    for start in range(0, len(records), 8):
        block = records[start:start + 8]
        prompt = ("Compare these recorded successful and failed training experiences. "
                  "Extract transferable action rules, correct misleading earlier rules, and keep "
                  "evidence-supported rules. Do not copy task-specific answers. Output the updated "
                  "numbered insight list.\n\nExisting insights:\n" + (insights or "(none)") +
                  "\n\nTraining experiences:\n" + "\n\n".join(
                      f"task_id={record['task_id']} success={record['success']} score={record['outcome']}\n" +
                      _experience_text(record) for record in block))
        output = llm("You learn general action insights from recorded training experiences.", prompt).strip()
        if not output:
            raise ValueError("ExpeL insight extraction returned no insights.")
        insights = output
        extraction_records.append({"task_ids": [item["task_id"] for item in block],
                                   "prompt": prompt, "output": output})
    artifact = {"method": "ExpeL", "source_url": PRIMARY_SOURCES["ExpeL"], "source_split": "train",
                "fidelity": "manuscript_adaptation_bm25_top5_with_offline_insight_extraction",
                "synthetic": synthetic, "experiences": records, "insights": insights,
                "extraction_records": extraction_records}
    with Path(out_path).open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return artifact


def _validate_expel_artifact(artifact_path, config):
    if artifact_path is None or not Path(artifact_path).is_file():
        raise FileNotFoundError("ExpeL requires a memory artifact built from actual training traces.")
    with open(artifact_path, encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("method") != "ExpeL" or artifact.get("source_split") != "train":
        raise ValueError("ExpeL artifact must identify the method and the train split.")
    if artifact.get("source_url") != PRIMARY_SOURCES["ExpeL"]:
        raise ValueError("ExpeL artifact must identify its primary source.")
    if artifact.get("fidelity") != "manuscript_adaptation_bm25_top5_with_offline_insight_extraction":
        raise ValueError("This runner requires the explicitly named BM25/top-5 ExpeL adaptation artifact.")
    if not artifact.get("insights") or not artifact.get("experiences") or not artifact.get("extraction_records"):
        raise ValueError("ExpeL requires both retrieved experiences and recorded LLM insight extraction.")
    experiences = artifact["experiences"]
    synthetic = artifact.get("synthetic") or any(item.get("synthetic") or
        item.get("backend") in {"synthetic", "mock"} for item in experiences)
    if synthetic and not config.mock:
        raise ValueError("Synthetic experience memory is allowed only for explicit mock runs.")
    for item in experiences:
        if (not item.get("task_id") or not item.get("task_goal") or item.get("split") != "train" or not item.get("steps") or
                not item.get("benchmark") or not isinstance(item.get("success"), bool)):
            raise ValueError("ExpeL experiences need benchmark/task IDs, train split and actual graded trajectories.")
        if any(not step.get("observation") or not step.get("action_taken") for step in item["steps"]):
            raise ValueError("ExpeL training experiences contain incomplete observations/actions.")
    tasks = {(item["benchmark"], item["task_id"]) for item in experiences}
    if not config.mock and len(tasks) < 50:
        raise ValueError("ExpeL requires 50 distinct training tasks under the manuscript protocol.")
    if sum(bool(item["success"]) for item in experiences) < (1 if config.mock else 5):
        raise ValueError("ExpeL retrieval needs at least five recorded successful training experiences.")
    if any(not entry.get("prompt") or not entry.get("output") or not entry.get("task_ids")
           for entry in artifact["extraction_records"]):
        raise ValueError("ExpeL insight extraction records require task IDs and complete LLM input/output.")
    return artifact


class BaselineAgent(Emotion2SkillAgent):
    def __init__(self, config: AgentConfig, method: str, llm=None, artifact_path=None,
                 max_reflections: int = 3):
        if method not in {"ZeroShot", "ReAct", "Reflexion", "ExpeL", "TextSimilarity"}:
            raise ValueError("This baseline requires a checked external implementation.")
        artifact = _validate_expel_artifact(artifact_path, config) if method == "ExpeL" else None
        super().__init__(config, llm=llm, validate_extractor=False)
        self.method_name, self.max_reflections = method, max_reflections
        self.baseline_metadata = {"method": method, "source_url": PRIMARY_SOURCES[method],
            "fidelity": "manuscript_protocol_adaptation", "max_reflections": max_reflections if method == "Reflexion" else 0}
        self._reflections, self._react_history = [], []
        self._memory = None
        self._retrieval = None
        if method == "ExpeL":
            experiences = artifact["experiences"]
            self._memory = artifact
            self._retrieval_records = [record for record in experiences if record.get("success")]
            if len(self._retrieval_records) < (1 if config.mock else 5):
                raise ValueError("ExpeL retrieval needs at least five recorded successful training experiences.")
            self._retrieval = BM25([_experience_text(record) for record in self._retrieval_records])
            self.baseline_metadata.update({"artifact_path": str(Path(artifact_path).resolve()),
                "artifact_sha256": hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest(),
                "fidelity": artifact.get("fidelity"), "source_task_ids": [item["task_id"] for item in experiences]})

    def load(self, *args, **kwargs):
        """Baselines use the shared LLM; no emotion extractor or encoder is loaded."""
        return None

    def _step(self, t, observation, context, library, history, use_emotion=False, collect_emotion=None):
        started = time.perf_counter()
        skill_name, scores, prompt, raw, thought = "", {}, "", "", ""
        if self.method_name == "TextSimilarity":
            if not library.skills:
                raise ValueError("TextSimilarity requires a nonempty skill library.")
            index = BM25([skill.description + "\n" + skill.sop for skill in library.skills])
            values = index.scores(context)
            scores = dict(zip(library.names(), values))
            skill_name = library.names()[max(range(len(values)), key=values.__getitem__)]
            skill = library[skill_name]
            action = self.executor.execute(observation, skill_name, skill.sop, history,
                                          context=context, action_instruction=self._action_instruction)
            prompt, raw = self.executor.last_prompt, self.executor.last_raw_output
        elif self.method_name == "ZeroShot":
            # No chain-of-thought request, candidate skills, SOPs, or retrieval.
            prompt = f"{context}\n\nEnvironment action syntax:\n{self._action_instruction}\n\nNext action:"
            raw = self.llm(ACTION_SYSTEM, prompt).strip()
            action = raw
        else:
            memories = ""
            if self._reflections:
                memories += "\n\nPrevious attempts' self-reflections:\n" + "\n".join(self._reflections[-3:])
            if self.method_name == "ExpeL":
                selected = self._retrieval.retrieve(context, 5)
                memories += "\n\nLearned training insights:\n" + self._memory["insights"]
                memories += "\n\nRetrieved training experiences:\n" + "\n\n".join(
                    _experience_text(self._retrieval_records[index]) for index in selected)
            trace = "\n".join(self._react_history)
            prompt = (f"{context}{memories}\n\nReasoning/action trace:\n{trace}\n\n"
                      f"Environment action syntax:\n{self._action_instruction}\n\n"
                      "Output 'Thought: <reasoning>' followed by exactly one 'Action: <next action>' line. "
                      "For a multiline code action, place a fenced code block after 'Action:'.")
            raw = self.llm("Reason about the current task, then choose the next environment action.", prompt).strip()
            thought, action = parse_react_action(raw)
            self._react_history.append(raw)
        if not action:
            raise ValueError(f"{self.method_name} returned an empty action.")
        record = StepRecord(t, observation, skill_name, -1, 0.0, [], action, 0.0, context=context,
            selection_raw_output=skill_name, candidate_scores=scores, action_prompt=prompt,
            action_raw_output=raw, decision_seconds=time.perf_counter() - started)
        return record, action

    def run_episode(self, env, library, *args, **kwargs):
        if self.method_name == "ExpeL":
            key = (getattr(env, "benchmark", self.config.benchmark), str(env.task_id))
            if key in {(item.get("benchmark"), str(item["task_id"])) for item in self._memory["experiences"]}:
                raise ValueError("Evaluation task overlaps the ExpeL training memory.")
        self._reflections, self._react_history = [], []
        kwargs["use_emotion"], kwargs["collect_emotion"] = False, False
        log = super().run_episode(env, library, *args, **kwargs)
        if self.method_name != "Reflexion":
            return log
        if log.benchmark == "mbpp":
            log.metadata.update({"retry_budget": 0, "total_attempts": 1, "metric": "pass_at_1"})
            return log
        attempts, reflection_prompts, reflections = [log.to_dict()], [], []
        for _ in range(self.max_reflections):
            if log.success:
                break
            prompt = (f"Task: {log.task_goal}\nOutcome: {log.outcome}; success: {log.success}\n"
                      "Recorded attempted trajectory:\n" + _experience_text(log.to_dict()) +
                      "\nExplain the failure and give a specific plan for the next attempt. "
                      "Use the environment feedback; do not invent observations.")
            reflection = self.llm("Reflect on a failed task attempt and improve the next attempt.", prompt).strip()
            if not reflection:
                raise ValueError("Reflexion returned an empty self-reflection.")
            self._reflections.append(reflection)
            reflection_prompts.append(prompt)
            reflections.append(reflection)
            initial, goal = env.reset(task_idx=log.task_id)
            self._react_history = []
            kwargs["initial_observation"], kwargs["task_goal"] = initial, goal
            log = super().run_episode(env, library, *args, **kwargs)
            attempts.append(log.to_dict())
        log.attempts, log.reflections, log.reflection_prompts = attempts, reflections, reflection_prompts
        log.metadata["retry_budget"] = self.max_reflections
        log.metadata["total_attempts"] = len(attempts)
        log.metadata["total_environment_steps"] = sum(item["n_steps"] for item in attempts)
        return log


class CheckedExternalAgent:
    """Require upstream adapters to preserve split boundaries and evidence schema."""
    def __init__(self, runner, manifest):
        self.runner, self.manifest = runner, manifest
        self.config = runner.config
        self.method_name = runner.method_name
        self.baseline_metadata = runner.baseline_metadata

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def load(self, *args, **kwargs):
        return self.runner.load(*args, **kwargs)

    def _check_task(self, env, task_id):
        if (env.benchmark == self.manifest["benchmark"] and str(task_id) in
                {str(value) for value in self.manifest["source_task_ids"]} and env.split != "train"):
            raise ValueError("External baseline evaluation overlaps its training artifact task IDs.")

    def run_episode(self, env, library, *args, **kwargs):
        self._check_task(env, env.task_id)
        return self.runner.run_episode(env, library, *args, **kwargs)

    def run_episodes(self, env, library, *args, **kwargs):
        if kwargs.get("evolution") and kwargs.get("phase") != "evolution":
            raise ValueError("External baseline updates are prohibited during final evaluation.")
        order = kwargs.get("task_order")
        if order is None:
            raise ValueError("External baselines must run an explicit fixed task_order.")
        for task_id in order:
            self._check_task(env, task_id)
        logs = self.runner.run_episodes(env, library, *args, **kwargs)
        if len(logs) != len(order):
            raise ValueError("External adapter did not return exactly one episode per declared task.")
        for log, task_id in zip(logs, order):
            if (not isinstance(log, EpisodeLog) or str(log.task_id) != str(task_id) or
                    log.backend != env.backend or log.benchmark != env.benchmark or log.split != env.split or
                    log.phase != kwargs.get("phase", "evaluation") or not log.steps or not log.task_goal):
                raise ValueError("External adapter did not preserve the common task/provenance/episode-log schema.")
        return logs


def _load_external(method, config, llm, artifact_path, external_adapter):
    if not external_adapter or not artifact_path:
        raise FileNotFoundError(f"{method} requires a genuine source-pinned external adapter and artifact manifest; "
                                "TextSimilarity is available as a separate lexical baseline.")
    manifest_path = Path(artifact_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    required = ("method", "source_url", "revision", "implementation_files", "artifacts", "protocol_config",
                "benchmark", "source_task_ids")
    if any(not manifest.get(key) for key in required):
        raise ValueError("External baseline manifest must include method/source/revision, file hashes, artifacts, "
                         "protocol_config and benchmark/train task IDs.")
    if manifest["method"] != method or manifest["source_url"] != PRIMARY_SOURCES[method]:
        raise ValueError("External baseline method/source does not match its cited primary paper.")
    if manifest.get("synthetic") and not config.mock:
        raise ValueError("Synthetic external artifacts require explicit mock execution.")
    if manifest.get("source_split") != "train":
        raise ValueError("External baseline skill artifacts must have a recorded train source split.")
    root = manifest_path.resolve().parent
    verified_implementations = set()
    for entry in manifest["implementation_files"] + manifest["artifacts"]:
        path = Path(entry["path"])
        path = path if path.is_absolute() else root / path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            raise ValueError(f"Missing or hash-mismatched external baseline file: {path}")
        if entry in manifest["implementation_files"]:
            verified_implementations.add(path.resolve())
    module_name, separator, factory_name = external_adapter.partition(":")
    if not separator:
        raise ValueError("external_adapter must use 'module:factory' syntax.")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    source_file = inspect.getsourcefile(factory)
    if source_file is None or Path(source_file).resolve() not in verified_implementations:
        raise ValueError("The actual adapter factory source must be in implementation_files with a valid hash.")
    llm = llm if llm is not None else Qwen3LLM(config.model_name_or_path, config.device,
                                           max_context_tokens=config.max_context_tokens, revision=config.model_revision)
    agent = factory(config=config, llm=llm, artifact_path=str(manifest_path.resolve()))
    declared = getattr(agent, "baseline_metadata", {})
    if declared.get("method") != method or declared.get("revision") != manifest["revision"]:
        raise ValueError("External adapter metadata does not match the pinned method/revision.")
    if not hasattr(agent, "config") or not all(callable(getattr(agent, name, None))
            for name in ("run_episode", "run_episodes", "load")):
        raise TypeError("External adapter must expose the common agent runner interface.")
    agent.method_name = method
    agent.baseline_metadata.update({"source_url": manifest["source_url"], "artifact_manifest": str(manifest_path.resolve()),
        "artifact_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "fidelity": "external_source_pinned_implementation; historical manuscript results not verified"})
    return CheckedExternalAgent(agent, manifest)


def build_baseline(method: str, config: AgentConfig, llm=None, artifact_path=None, external_adapter=None):
    if method not in PRIMARY_SOURCES:
        raise ValueError(f"Unknown baseline {method}; choices: {list(PRIMARY_SOURCES)}")
    if method in {"MASA", "EvoSkill", "Trace2Skill"}:
        # Validate before loading an expensive backbone when the adapter is absent.
        if not external_adapter or not artifact_path:
            return _load_external(method, config, llm, artifact_path, external_adapter)
        return _load_external(method, config, llm, artifact_path, external_adapter)
    return BaselineAgent(config, method, llm=llm, artifact_path=artifact_path)


def load_episode_logs(path) -> list[EpisodeLog]:
    """Load the complete shared JSONL schema for offline experience learning."""
    episodes = []
    episode_fields = {item.name for item in dataclasses.fields(EpisodeLog)}
    step_fields = {item.name for item in dataclasses.fields(StepRecord)}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            steps = [StepRecord(**{key: value for key, value in step.items() if key in step_fields})
                     for step in record["steps"]]
            episodes.append(EpisodeLog(**{key: value for key, value in record.items()
                                          if key in episode_fields and key != "steps"}, steps=steps))
    return episodes


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Build ExpeL insights from actual training episode logs.")
    parser.add_argument("command", choices=["build-memory"])
    parser.add_argument("--logs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    episodes = load_episode_logs(args.logs)
    if Path(args.out).exists():
        raise FileExistsError("ExpeL output already exists; keep the recorded memory immutable.")
    if (any(episode.split != "train" or episode.backend in {"mock", "synthetic"} for episode in episodes) or
            len({(episode.benchmark, episode.task_id) for episode in episodes}) < 50):
        raise ValueError("Provide real, explicitly train-split logs from at least 50 distinct training tasks.")
    llm = Qwen3LLM(args.model, args.device)
    build_expel_memory(llm, episodes, args.out)


if __name__ == "__main__":
    main()
