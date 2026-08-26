from __future__ import annotations
import json
from pathlib import Path


def load_config(root, dataset):
    root = Path(root)
    name = dataset.lower().replace("-", "").replace(".", "") + ".json"
    path = root / "configs" / "inductive" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
