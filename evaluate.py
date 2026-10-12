"""
Evaluate Emotion2Skill
=======================
Runs evaluation on WebShop or ALFWorld and prints results matching
Table 1 of the paper.

Usage
-----
# Full Emotion2Skill (Qwen3-8B, WebShop)
python evaluate.py \
    --model    Qwen/Qwen3-8B \
    --env      webshop \
    --extractor checkpoints/extractor_qwen3_8b.pkl \
    --encoder   checkpoints/encoder_webshop.pt \
    --tau       0.3 \
    --evolution \
    --n_episodes 500

# ALFWorld (134 test episodes)
python evaluate.py \
    --model    Qwen/Qwen3-8B \
    --env      alfworld \
    --extractor checkpoints/extractor_qwen3_8b.pkl \
    --encoder   checkpoints/encoder_alfworld.pt \
    --n_episodes 134

# Ablation: no emotion extraction (plain within-class means)
python evaluate.py --env webshop --ablation no_extraction

# Ablation: no encoder (top-3 raw labels injected)
python evaluate.py --env webshop --ablation no_encoder

# Ablation: selection only (no evolution)
python evaluate.py --env webshop --ablation selection_only

# Tau sensitivity sweep
python evaluate.py --env webshop --tau_sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional

from emotion2skill.agent import AgentConfig, Emotion2SkillAgent
from emotion2skill.envs.alfworld import ALFWorldEnv
from emotion2skill.envs.webshop import WebShopEnv
from emotion2skill.skills import get_library


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Emotion2Skill")

    # Model
    p.add_argument("--model",       type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--model_key",   type=str, default="qwen3-8b",
                   choices=["qwen3-8b", "qwen3-14b"])
    p.add_argument("--device",      type=str, default="cuda")

    # Environment
    p.add_argument("--env",         type=str, default="webshop",
                   choices=["webshop", "alfworld"])
    p.add_argument("--n_episodes",  type=int, default=None,
                   help="Override default episode count")
    p.add_argument("--n_runs",      type=int, default=3,
                   help="Number of independent runs for std-dev reporting")

    # Checkpoints
    p.add_argument("--extractor",   type=str, default=None)
    p.add_argument("--encoder",     type=str, default=None)

    # Hyper-parameters
    p.add_argument("--tau",         type=float, default=0.3)
    p.add_argument("--eta",         type=float, default=0.5)
    p.add_argument("--evolution",   action="store_true",
                   help="Enable skill evolution")
    p.add_argument("--evolution_rounds", type=int, default=5)
    p.add_argument("--episode_per_round", type=int, default=20)

    # Ablation mode
    p.add_argument("--ablation",    type=str, default=None,
                   choices=[
                       "no_extraction",   # plain within-class means
                       "no_encoder",      # top-3 raw labels
                       "selection_only",  # no evolution
                       "no_diagnostics",  # outcome-based evolution only
                       "full",            # default (same as no flag)
                   ])

    # Special modes
    p.add_argument("--tau_sweep",   action="store_true",
                   help="Sweep τ ∈ {0.1, 0.3, 0.6, 0.9}")

    # Output
    p.add_argument("--out_dir",     type=str, default="results")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_EPISODES = {"webshop": 500, "alfworld": 134}


def build_agent(args: argparse.Namespace, tau: float) -> Emotion2SkillAgent:
    config = AgentConfig(
        model_name_or_path  = args.model,
        model_key           = args.model_key,
        device              = args.device,
        extractor_path      = args.extractor,
        encoder_path        = args.encoder,
        tau                 = tau,
        eta                 = args.eta,
        evolution_rounds    = args.evolution_rounds,
        log_dir             = str(Path(args.out_dir) / "logs"),
    )
    agent = Emotion2SkillAgent(config)
    agent.load()
    return agent


def build_env(env_name: str, max_steps: int = 50):
    if env_name == "webshop":
        return WebShopEnv(split="test", max_steps=15)
    elif env_name == "alfworld":
        return ALFWorldEnv(max_steps=max_steps)
    else:
        raise ValueError(f"Unknown env: {env_name}")


def run_single(
    agent:       Emotion2SkillAgent,
    env,
    env_name:    str,
    library,
    n_episodes:  int,
    ablation:    Optional[str],
    episode_per_round: int,
    verbose:     bool,
) -> dict:
    """Run one evaluation pass and return metric dict."""
    use_emotion = ablation not in ("no_extraction", "no_encoder")
    evolution   = ablation not in ("selection_only",)

    # Patch encoder for specific ablations
    if ablation == "no_encoder":
        # Inject top-3 emotion labels as raw text (bypass encoder)
        _patch_no_encoder(agent)
    elif ablation == "no_diagnostics":
        # Use outcome-only evolution (no emotion diagnostics)
        _patch_no_diagnostics(agent)

    fmt_fn = None
    if env_name == "webshop":
        fmt_fn = lambda tg, h, o: env.format_context(tg, h, o)
    elif env_name == "alfworld":
        fmt_fn = lambda tg, h, o: env.format_context(tg, h, o)

    logs = agent.run_episodes(
        env              = env,
        library          = library,
        n_episodes       = n_episodes,
        use_emotion      = use_emotion,
        evolution        = evolution,
        episode_per_round = episode_per_round,
        verbose          = verbose,
        format_context_fn = fmt_fn,
    )

    if env_name == "webshop":
        scores  = [log.outcome for log in logs]
        return WebShopEnv.compute_metrics(scores)
    else:
        results = []
        for log in logs:
            # Recover task_type from episode_id or log metadata
            tt = getattr(env, "_task_type", None) or "pick"
            results.append({"task_type": tt, "success": log.success})
        return ALFWorldEnv.compute_metrics(results)


# ── Ablation patches ──────────────────────────────────────────────────────────

def _patch_no_encoder(agent: Emotion2SkillAgent) -> None:
    """Replace encoder.encode() with a top-3 raw-label injection."""
    from emotion2skill.encoder import EMOTION_TEMPLATES, EmotionSummary
    from emotion2skill.extraction import GOEMOTIONS_LABELS
    import torch

    original_encode = agent.encoder.encode

    def top3_encode(e_t: "torch.Tensor") -> EmotionSummary:
        # Top-3 labels as text, no MLP
        vals, idxs = torch.topk(e_t, 3)
        label_str  = ", ".join(GOEMOTIONS_LABELS[i] for i in idxs.tolist())
        return EmotionSummary(
            template_idx  = 0,
            template_text = f"Top-3 active states: {label_str}",
            confidence    = 0.99,   # always admitted (ablation)
            emotion_state = e_t,
            is_admitted   = True,
        )

    agent.encoder.encode = top3_encode


def _patch_no_diagnostics(agent: Emotion2SkillAgent) -> None:
    """Replace evolver to use outcome-only prompts (no diagnostic windows)."""
    from emotion2skill.evolution import (
        EpisodeLog, build_evolution_prompt, EVOLUTION_SYSTEM
    )

    original_run_round = agent.evolver.run_round

    def outcome_only_run_round(library, episodes, verbose=False):
        for ep in episodes:
            agent.evolver.tracker.record_episode(ep)
        under = agent.evolver.tracker.underperforming_skills()
        rewrites = {}
        for skill_name in under:
            try:
                skill = library[skill_name]
            except KeyError:
                continue
            # Build prompt with empty diagnostic windows
            last_ep = next((e for e in reversed(episodes) if not e.success), episodes[-1])
            prompt  = build_evolution_prompt(skill, [], last_ep.outcome, last_ep.success)
            new_sop = agent.evolver._llm(EVOLUTION_SYSTEM, prompt).strip()
            if new_sop:
                from emotion2skill.selection import Skill
                library.replace(skill_name, Skill(skill.name, new_sop, skill.description))
                rewrites[skill_name] = new_sop
        return rewrites

    agent.evolver.run_round = outcome_only_run_round


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def report_webshop(metrics_runs: list[dict]) -> None:
    scores  = [m["score"]        for m in metrics_runs]
    successes = [m["success_rate"] for m in metrics_runs]
    print(f"\n{'='*50}")
    print(f"WebShop Results ({len(metrics_runs)} run(s))")
    print(f"  Score:   {statistics.mean(scores):.1f}  "
          f"(std={statistics.pstdev(scores):.1f})")
    print(f"  Succ.%:  {statistics.mean(successes):.1f}  "
          f"(std={statistics.pstdev(successes):.1f})")
    print(f"{'='*50}\n")


def report_alfworld(metrics_runs: list[dict]) -> None:
    from emotion2skill.envs.alfworld import TASK_TYPES
    print(f"\n{'='*70}")
    print(f"ALFWorld Results ({len(metrics_runs)} run(s))")
    header = f"{'Task':12s}" + "".join(f"{tt:10s}" for tt in TASK_TYPES) + f"{'Avg':10s}"
    print(f"  {header}")

    means = {}
    stds  = {}
    for tt in TASK_TYPES + ["avg"]:
        vals      = [m.get(tt, 0.0) for m in metrics_runs]
        means[tt] = statistics.mean(vals)
        stds[tt]  = statistics.pstdev(vals)

    row_m = f"  {'Mean':12s}" + "".join(
        f"{means[tt]:10.1f}" for tt in TASK_TYPES
    ) + f"{means['avg']:10.1f}"
    row_s = f"  {'Std':12s}" + "".join(
        f"{stds[tt]:10.1f}" for tt in TASK_TYPES
    ) + f"{stds['avg']:10.1f}"
    print(row_m)
    print(row_s)
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    n_episodes = args.n_episodes or DEFAULT_N_EPISODES[args.env]
    ablation   = args.ablation or "full"

    # ── τ sensitivity sweep ────────────────────────────────────────────────────
    if args.tau_sweep:
        tau_grid = [0.1, 0.3, 0.6, 0.9]
        sweep_results = {}
        for tau in tau_grid:
            print(f"\n[τ sweep] τ = {tau}")
            agent   = build_agent(args, tau)
            env     = build_env(args.env)
            library = get_library(args.env)
            metrics = run_single(
                agent, env, args.env, library,
                n_episodes, ablation, args.episode_per_round, args.verbose
            )
            sweep_results[str(tau)] = metrics
            print(f"  τ={tau}: {metrics}")

        out_path = Path(args.out_dir) / f"tau_sweep_{args.env}.json"
        with open(out_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nτ sweep saved to {out_path}")
        return

    # ── standard multi-run evaluation ─────────────────────────────────────────
    metrics_runs = []
    for run_id in range(args.n_runs):
        print(f"\n[Run {run_id + 1}/{args.n_runs}] env={args.env} "
              f"ablation={ablation} τ={args.tau}")
        agent   = build_agent(args, args.tau)
        env     = build_env(args.env)
        library = get_library(args.env)

        metrics = run_single(
            agent, env, args.env, library,
            n_episodes, ablation, args.episode_per_round, args.verbose
        )
        metrics_runs.append(metrics)
        print(f"  Run {run_id + 1} metrics: {metrics}")

    # ── report ────────────────────────────────────────────────────────────────
    if args.env == "webshop":
        report_webshop(metrics_runs)
    else:
        report_alfworld(metrics_runs)

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out_dir) / f"eval_{args.env}_{ablation}.json"
    with open(out_path, "w") as f:
        json.dump(metrics_runs, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
