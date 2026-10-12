#!/usr/bin/env python3
"""Offline build with validation-only layer selection and separate held-out test."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion2skill.config import resolve_config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--model_key")
    parser.add_argument("--model_revision")
    parser.add_argument("--device")
    parser.add_argument("--out")
    args = parser.parse_args()
    config = resolve_config(args.config, {"model.name_or_path": args.model,
        "model.key": args.model_key, "model.revision": args.model_revision,
        "model.device": args.device, "extractor.checkpoint": args.out})
    from emotion2skill.extraction import EmotionExtractor
    settings = config["extractor"]
    for key in ("goemotions_validation", "goemotions_test"):
        if not Path(settings[key]).is_file():
            raise FileNotFoundError(f"Prepare official GoEmotions splits before extractor build: {settings[key]}")
    model = config["model"]
    extractor = EmotionExtractor(model["name_or_path"], model["key"], model["device"], revision=model.get("revision"))
    extractor.build(stories_path=settings["stories_path"],
        goemotions_validation_path=settings["goemotions_validation"],
        goemotions_test_path=settings["goemotions_test"], save_path=settings["checkpoint"])
if __name__ == "__main__":
    main()
