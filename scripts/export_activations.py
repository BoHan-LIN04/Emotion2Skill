#!/usr/bin/env python3
"""Export held-out activations for layer and representation analysis."""
import argparse
from emotion2skill.config import resolve_config
from emotion2skill.extraction import EmotionExtractor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--layer", type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    config = resolve_config(args.config)
    model, settings = config["model"], config["extractor"]
    extractor = EmotionExtractor.load(settings["checkpoint"], model["name_or_path"], model["device"],
                                     revision=model.get("revision"))
    probe = settings["goemotions_validation"] if args.split == "validation" else settings["goemotions_test"]
    extractor.export_probe_activations(probe, args.out, split=args.split, layer_idx=args.layer)


if __name__ == "__main__":
    main()
