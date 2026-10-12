"""Collect train-split warm-up states and train the emotion encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from emotion2skill.agent import AgentConfig, Emotion2SkillAgent, validate_inference_assets
from emotion2skill.encoder import (
    C, K, EMOTION_TEMPLATES, TEMPLATE_BANK_VERSION, TEMPLATE_EMOTION_WEIGHTS,
    EmotionEncoderMLP, cluster_template_mapping, kmeans_pseudo_labels,
    supervised_contrastive_loss, template_classification_loss,
)
from emotion2skill.skills import get_library


def collect_warmup_data(agent, env, library, n_episodes=250, verbose=False,
                        format_context_fn=None, *, namespace="warmup",
                        skill_offset=0, min_signal=0.05, return_metadata=False,
                        episode_order=None):
    """Collect emotion states without prompt injection and filter low-signal steps.

    return_metadata includes episode groups and extraction prefixes for PCA fitting.
    """
    if n_episodes <= 0:
        raise ValueError("Warm-up episode count must be positive")
    skill2idx = {name: i + skill_offset for i, name in enumerate(library.names())}
    kwargs = dict(env=env, library=library, n_episodes=n_episodes,
                  use_emotion=False, collect_emotion=True, evolution=False,
                  verbose=verbose, format_context_fn=format_context_fn, phase="warmup")
    if episode_order is not None:
        kwargs["task_order"] = episode_order
    logs = agent.run_episodes(**kwargs)
    e_list, s_list, y_list, groups, prompts, tasks = [], [], [], [], [], []
    episode_task_ids = [str(getattr(log, "task_id", i)) for i, log in enumerate(logs)]
    filtered = 0
    for ep_idx, ep_log in enumerate(logs):
        for step in ep_log.steps:
            e = step.emotion_state.detach().float().cpu()
            if e.shape != (K,) or not torch.isfinite(e).all():
                raise ValueError("Warm-up recorded an invalid emotion vector")
            if e.max().item() < min_signal:
                filtered += 1
                continue
            if step.skill_selected not in skill2idx:
                raise ValueError(f"Unknown warm-up skill: {step.skill_selected}")
            e_list.append(e)
            s_list.append(skill2idx[step.skill_selected])
            y_list.append(bool(ep_log.success))
            groups.append(f"{namespace}:{getattr(ep_log, 'episode_id', ep_idx)}")
            prompts.append(getattr(step, "extraction_prompt", ""))
            tasks.append(str(getattr(ep_log, "task_id", ep_idx)))
    if not e_list:
        raise RuntimeError("No informative warm-up states; check extraction and environments")
    tensors = (torch.stack(e_list), torch.tensor(s_list, dtype=torch.long),
               torch.tensor(y_list, dtype=torch.bool))
    metadata = {"group_ids": groups, "extraction_prompts": prompts,
                "task_ids": tasks, "episodes": len(logs), "filtered_steps": filtered,
                "episode_task_ids": episode_task_ids,
                "benchmark": getattr(env, "benchmark", namespace),
                "retained_steps": len(e_list), "namespace": namespace,
                "skill_labels": skill2idx, "min_signal": min_signal}
    if callable(getattr(env, "metadata", None)):
        metadata["environment"] = env.metadata()
        metadata["manifest_sha256"] = metadata["environment"].get("manifest_sha256")
    return (*tensors, metadata) if return_metadata else tensors


def split_training_indices(n, *, group_ids=None, validation_fraction=0.2, seed=42):
    """Partition complete episodes when group IDs are available."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if group_ids is None:
        group_ids = [str(i) for i in range(n)]
    if len(group_ids) != n:
        raise ValueError("group_ids must match the feature count")
    unique = list(dict.fromkeys(group_ids))
    if len(unique) < 2:
        raise ValueError("Need at least two episodes/groups for a held-out validation set")
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    val_groups = set(unique[:n_val])
    train = torch.tensor([i for i, group in enumerate(group_ids) if group not in val_groups])
    val = torch.tensor([i for i, group in enumerate(group_ids) if group in val_groups])
    if not len(train) or not len(val):
        raise ValueError("Training and validation partitions must both be nonempty")
    return train.long(), val.long()


def train_encoder(e_vectors, skill_labels, success_flags, out_path, *, epochs=50,
                  batch_size=64, lr=1e-3, weight_decay=1e-4, patience=5,
                  kappa=0.07, kappa_prime=0.10, lam=0.5, confidence_weight=1.0,
                  device="cuda", seed=42, group_ids=None, validation_fraction=0.2,
                  semantic_vectors=None, extractor_mode="contrastive",
                  shuffle_context=False, source_metadata=None):
    """Train encoder heads and save the semantic mapping and training metadata.

    Aligned semantic_vectors interpret clusters formed from generic-PCA features.
    shuffle_context permutes skill/outcome tuples only within the training partition,
    keeping input vectors and template semantics fixed.
    """
    if epochs < 1 or batch_size < 1 or patience < 1 or confidence_weight <= 0:
        raise ValueError("Positive epochs, batch size, patience and BCE weight are required")
    if extractor_mode not in {"contrastive", "class_mean", "generic_pca"}:
        raise ValueError("Unknown extractor mode")
    e_vectors = e_vectors.detach().float().cpu()
    skill_labels = skill_labels.detach().long().cpu()
    success_flags = success_flags.detach().bool().cpu()
    n = len(e_vectors)
    if e_vectors.shape != (n, K) or n < C + 1 or not torch.isfinite(e_vectors).all():
        raise ValueError("Need enough finite (N, 27) training features")
    if skill_labels.shape != (n,) or success_flags.shape != (n,):
        raise ValueError("Skill/outcome labels must match the feature count")
    if semantic_vectors is None:
        if extractor_mode == "generic_pca":
            raise ValueError("Generic PCA requires aligned GoEmotions semantic vectors")
        semantic_vectors = e_vectors
    semantic_vectors = semantic_vectors.detach().float().cpu()
    if semantic_vectors.shape != e_vectors.shape or not torch.isfinite(semantic_vectors).all():
        raise ValueError("Aligned semantic features must be finite (N, 27)")
    torch.manual_seed(seed)
    random.seed(seed)
    train_idx, val_idx = split_training_indices(
        n, group_ids=group_ids, validation_fraction=validation_fraction, seed=seed)
    cluster_labels, centroids = kmeans_pseudo_labels(
        e_vectors[train_idx], n_clusters=C, seed=seed, return_centroids=True)
    semantic_centroids = torch.stack([
        semantic_vectors[train_idx][cluster_labels == j].mean(0)
        if (cluster_labels == j).any() else semantic_vectors[train_idx].mean(0)
        for j in range(C)
    ])
    mapping, alignment_scores = cluster_template_mapping(semantic_centroids)
    pseudo_train = mapping[cluster_labels]
    pseudo_val = mapping[torch.cdist(e_vectors[val_idx], centroids).argmin(1)]
    sk_train, y_train = skill_labels[train_idx], success_flags[train_idx]
    permutation = torch.arange(len(train_idx))
    if shuffle_context:
        permutation = torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(seed + 1))
        sk_train, y_train = sk_train[permutation], y_train[permutation]
    dataset = TensorDataset(e_vectors[train_idx], sk_train, y_train, pseudo_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(seed))
    if len(loader) == 0:
        raise ValueError("Training loader has zero batches")
    model = EmotionEncoderMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best_state, best_loss, stale, updates = [], None, float("inf"), 0, 0

    def losses(e, sk, suc, pseudo):
        r, confidence = model(e)
        ctr = supervised_contrastive_loss(r, sk, suc, temperature=kappa)
        cls = template_classification_loss(r, pseudo, model.prototypes, temperature=kappa_prime)
        bce = F.binary_cross_entropy(confidence, suc.float())
        return ctr + lam * cls + confidence_weight * bce, ctr, cls, bce

    validation = tuple(t.to(device) for t in (e_vectors[val_idx], skill_labels[val_idx],
                                             success_flags[val_idx], pseudo_val))
    for epoch in range(1, epochs + 1):
        model.train()
        sums = [0.0] * 4
        seen = 0
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            terms = losses(*batch)
            if not torch.isfinite(terms[0]):
                raise RuntimeError("Nonfinite encoder loss")
            optimizer.zero_grad(set_to_none=True)
            terms[0].backward()
            if model.w_c.weight.grad is None or model.w_c.bias.grad is None:
                raise RuntimeError("Confidence head is disconnected from training loss")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(batch[0])
            sums = [total + term.item() * count for total, term in zip(sums, terms)]
            seen += count
            updates += 1
        if seen == 0:
            raise RuntimeError("No optimizer updates occurred")
        model.eval()
        with torch.no_grad():
            val_terms = losses(*validation)
            if not all(torch.isfinite(term) for term in val_terms):
                raise RuntimeError("Nonfinite validation loss")
            val_loss = val_terms[0].item()
        row = {"epoch": epoch, "loss": sums[0] / seen, "ctr": sums[1] / seen,
               "cls": sums[2] / seen, "bce": sums[3] / seen,
               "validation_loss": val_loss, "validation_bce": val_terms[3].item()}
        history.append(row)
        print(f"Epoch {epoch}/{epochs}: loss={row['loss']:.4f}, val={val_loss:.4f}, BCE={row['bce']:.4f}")
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, confidence = model(validation[0])
        confidence_metrics = {"validation_bce": F.binary_cross_entropy(
            confidence, validation[2].float()).item(),
            "validation_brier": ((confidence - validation[2].float()) ** 2).mean().item()}
    metadata = {
        "trained": True, "confidence_trained": True, "optimizer_updates": updates,
        "extractor_mode": extractor_mode, "shuffle_context": bool(shuffle_context),
        "template_bank_version": TEMPLATE_BANK_VERSION, "template_bank": EMOTION_TEMPLATES,
        "template_semantic_weights": TEMPLATE_EMOTION_WEIGHTS,
        "cluster_to_template": mapping.tolist(), "semantic_mapping_method": "GoEmotions centroid/profile Hungarian assignment",
        "cluster_centroids": centroids.tolist(), "semantic_centroids": semantic_centroids.tolist(),
        "semantic_alignment_scores": alignment_scores.tolist(),
        "train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist(),
        "context_permutation": permutation.tolist() if shuffle_context else None,
        "split_unit": "episode" if group_ids is not None else "record",
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "confidence_metrics": confidence_metrics,
        "hyperparameters": {"seed": seed, "optimizer": "AdamW", "lr": lr,
            "weight_decay": weight_decay, "batch_size": batch_size, "max_epochs": epochs,
            "patience": patience, "kappa": kappa, "kappa_prime": kappa_prime,
            "lambda": lam, "confidence_weight": confidence_weight,
            "validation_fraction": validation_fraction},
        "source": source_metadata or {},
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "state_dict": best_state, "metadata": metadata}, output)
    output.with_name(output.stem + "_history.json").write_text(
        json.dumps({"history": history, "metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
    model.training_metadata = metadata
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--model_key", choices=["qwen3-8b", "qwen3-14b"], default=None)
    p.add_argument("--env", choices=["mixed", "webshop", "alfworld"], default="mixed")
    p.add_argument("--warmup", type=int, default=250, help="Episodes per environment")
    p.add_argument("--webshop_manifest", default="manifests/webshop.json")
    p.add_argument("--alfworld_manifest", default="manifests/alfworld.json")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--confidence_weight", type=float, default=1.0)
    p.add_argument("--min_signal", type=float, default=0.05)
    p.add_argument("--validation_fraction", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_context_tokens", type=int, default=None)
    p.add_argument("--extractor", default=None)
    p.add_argument("--extractor_out", default=None, help="PCA/control extractor checkpoint output")
    p.add_argument("--feature_mode", choices=["contrastive", "class_mean", "generic_pca"], default="contrastive")
    p.add_argument("--shuffle_context", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out", default="checkpoints/encoder.pt")
    p.add_argument("--log_dir", default="logs/warmup")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--config", default=None, help="YAML training/model/path overrides")
    p.add_argument("--model_revision", default=None, help="Immutable backbone revision; may also come from YAML")
    args = p.parse_args()
    from emotion2skill.config import resolve_config
    keys = {"model": "model.name_or_path", "model_key": "model.key", "model_revision": "model.revision", "device": "model.device",
            "max_context_tokens": "model.max_context_tokens",
            "env": "training.env", "warmup": "training.warmup_episodes_per_env",
            "webshop_manifest": "datasets.webshop.manifest", "alfworld_manifest": "datasets.alfworld.manifest",
            "extractor": "extractor.checkpoint", "extractor_out": "training.extractor_out",
            "feature_mode": "extractor.mode", "shuffle_context": "training.shuffle_context",
            "out": "encoder.checkpoint", "tau": "encoder.tau", "lam": "encoder.lambda",
            "log_dir": "logging.warmup_dir",
            **{name: "training." + name for name in ("epochs", "batch_size", "lr", "weight_decay",
                "patience", "confidence_weight", "min_signal", "validation_fraction", "seed")}}
    option_destinations = {option: action.dest for action in p._actions for option in action.option_strings}
    explicit = {option_destinations[token.split("=", 1)[0]] for token in sys.argv[1:]
                if token.split("=", 1)[0] in option_destinations}
    config = resolve_config(args.config, {keys[name]: getattr(args, name)
                                          for name in explicit if name in keys})
    for name, dotted in keys.items():
        value = config
        for key in dotted.split("."):
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            setattr(args, name, value)
    if not args.extractor:
        p.error("Provide --extractor or extractor.checkpoint in --config")
    if args.max_context_tokens is None:
        args.max_context_tokens = 8192 if args.model_key == "qwen3-14b" else 4096
    args.effective_config = config
    return args


def main():
    args = parse_args()
    from emotion2skill.envs import build_env
    from emotion2skill.envs.common import load_manifest
    names = ["webshop", "alfworld"] if args.env == "mixed" else [args.env]
    for name in names:
        manifest = load_manifest(getattr(args, name + "_manifest"), name)
        if len(manifest["splits"]["train"]) < args.warmup:
            raise ValueError(f"{name} train manifest has fewer than {args.warmup} warm-up tasks")
    initial_mode = "contrastive" if args.feature_mode == "generic_pca" else args.feature_mode
    config = AgentConfig(model_name_or_path=args.model, model_key=args.model_key,
                         model_revision=args.effective_config["model"].get("revision"),
                         device=args.device, extractor_path=args.extractor,
                         encoder_path=None, extractor_mode=initial_mode, tau=args.tau,
                         log_dir=args.log_dir, seed=args.seed, max_context_tokens=args.max_context_tokens)
    validate_inference_assets(config, require_encoder=False)
    agent = Emotion2SkillAgent(config)
    agent.load(require_encoder=False)
    all_e, all_s, all_y, metadata_list = [], [], [], []
    offset = 0
    for name in names:
        manifest = getattr(args, name + "_manifest")
        environment_options = {}
        if name == "alfworld":
            settings = args.effective_config.get("datasets", {}).get(name, {})
            environment_options = {key: settings[key] for key in ("data_path", "config_path") if key in settings}
        env = build_env(name, manifest_path=manifest, split="train", mock=False,
                        max_steps=15 if name == "webshop" else 50, **environment_options)
        library = get_library(name)
        data = collect_warmup_data(agent, env, library, n_episodes=args.warmup,
                                   verbose=args.verbose, namespace=name, skill_offset=offset,
                                   min_signal=args.min_signal, return_metadata=True)
        e, skills, outcomes, metadata = data
        metadata["manifest"] = str(Path(manifest).resolve())
        metadata["manifest_sha256"] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        all_e.append(e); all_s.append(skills); all_y.append(outcomes); metadata_list.append(metadata)
        offset += len(library)
        if hasattr(env, "close"):
            env.close()
    semantic_vectors = torch.cat(all_e)
    features = semantic_vectors.clone()
    groups = [g for entry in metadata_list for g in entry["group_ids"]]
    if args.feature_mode == "generic_pca":
        prompts = [s for entry in metadata_list for s in entry["extraction_prompts"]]
        if not all(prompts):
            raise RuntimeError("Warm-up logs must retain complete extraction prompts for PCA")
        train_idx, _ = split_training_indices(len(prompts), group_ids=groups,
            validation_fraction=args.validation_fraction, seed=args.seed)
        agent.extractor.fit_generic_pca([prompts[i] for i in train_idx.tolist()], split="train")
        agent.extractor = agent.extractor.with_mode("generic_pca")
        features = torch.stack([agent.extractor.get_emotion_state(prompt) for prompt in prompts])
        extractor_out = args.extractor_out or str(Path(args.out).with_suffix(".extractor.pkl"))
        agent.extractor.save(extractor_out)
    else:
        extractor_out = args.extractor
    source_metadata = {"warmup": metadata_list, "extractor_path": str(Path(extractor_out).resolve()),
                       "extractor_sha256": hashlib.sha256(Path(extractor_out).read_bytes()).hexdigest(),
                       "warmup_policy": "ReAct with shared skill library; confidence/emotion withheld; collect_emotion=True",
                       "environment_split": "train",
                       "effective_config": args.effective_config}
    tuples_path = Path(args.out).with_name(Path(args.out).stem + "_warmup.pt")
    tuples_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "semantic_vectors": semantic_vectors,
                "skill_labels": torch.cat(all_s), "success_flags": torch.cat(all_y),
                "group_ids": groups, "metadata": source_metadata}, tuples_path)
    source_metadata["warmup_tuples_path"] = str(tuples_path.resolve())
    train_encoder(features, torch.cat(all_s), torch.cat(all_y), args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, lam=args.lam,
        confidence_weight=args.confidence_weight, device=args.device, seed=args.seed,
        group_ids=groups, validation_fraction=args.validation_fraction,
        semantic_vectors=semantic_vectors, extractor_mode=args.feature_mode,
        shuffle_context=args.shuffle_context, source_metadata=source_metadata)


if __name__ == "__main__":
    main()
