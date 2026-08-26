#!/usr/bin/env python3
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
print("Inductive (bundled raw splits):")
for ds in ("FB15k-237", "WN18RR", "NELL-995"):
    versions = [p.name for p in sorted((ROOT / "datasets" / "inductive" / ds).glob("v*")) if p.is_dir()]
    print(f"  {ds}: {', '.join(versions)}")
