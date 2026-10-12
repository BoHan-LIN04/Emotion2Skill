"""Content identity for local backbone and tokenizer assets."""
import hashlib
import json
from pathlib import Path


def local_model_identity(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return None
    names = ("*.safetensors", "pytorch_model*.bin", "*.index.json", "config.json",
             "generation_config.json", "tokenizer*", "vocab*", "merges.txt",
             "special_tokens_map.json", "added_tokens.json", "*.jinja", "*.py")
    files = sorted({file for pattern in names for file in root.glob(pattern) if file.is_file()})
    if not any(file.suffix == ".safetensors" or file.name.startswith("pytorch_model") and file.suffix == ".bin"
               for file in files):
        raise ValueError("A local model identity requires actual backbone weight files.")
    records = []
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records.append({"name": file.relative_to(root).as_posix(), "size": file.stat().st_size,
                        "sha256": digest.hexdigest()})
    manifest = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"algorithm": "sha256-local-backbone-tokenizer-v1", "digest": hashlib.sha256(manifest).hexdigest(),
            "files": records}
