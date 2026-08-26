#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cqr_mass.data import load_dataset
from cqr_mass.evaluation import (
    average_tie_ranks,
    metrics,
    ordinal_tie_ranks,
    pessimistic_tie_ranks,
)
from cqr_mass.model import CQRMassAStar


def build(data, cfg):
    return CQRMassAStar(
        data, dim=cfg["entity_dim"], cqdim=cfg["cqr_dim"], q_layers=cfg["cqr_layers"],
        num_layer=cfg["astar_layers"], node_ratio=cfg["train_node_ratio"],
        test_node_ratio=cfg["test_node_ratio"], late_fusion_max_ratio=cfg["late_fusion_max_ratio"],
        mass_init_alpha=cfg["mass_init_alpha"],
        mass_fixed_alpha=cfg.get("mass_fixed_alpha"),
        mass_support_gate=cfg.get("mass_support_gate", "none"),
    )


def main():
    p = argparse.ArgumentParser(description="Filtered full-entity evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--save-ranks")
    p.add_argument(
        "--tie-policy",
        choices=["average", "ordinal", "pessimistic"],
        default="average",
    )
    p.add_argument(
        "--mass-support-gate",
        choices=[
            "none",
            "binary",
            "one_sided",
            "q2_only",
            "q3_only",
            "directional",
            "evidence_consistent",
        ],
    )
    args = p.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA is not available.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    if args.mass_support_gate is not None:
        cfg["mass_support_gate"] = args.mass_support_gate
    data = load_dataset(
        ROOT,
        ckpt["dataset"],
        ckpt.get("version", "v1"),
        cfg["batch_size"],
    )
    model = build(data, cfg).to(torch.device(args.device))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    n = data.n_valid if args.split == "valid" else data.n_test
    mode = data.mode_for_split(args.split)
    bs = args.batch_size or cfg.get("eval_batch_size", cfg["batch_size"])
    ranker = {
        "average": average_tie_ranks,
        "ordinal": ordinal_tie_ranks,
        "pessimistic": pessimistic_tie_ranks,
    }[args.tie_policy]
    all_ranks = []
    with torch.inference_mode():
        for st in range(0, n, bs):
            idx = np.arange(st, min(st + bs, n))
            heads, rels, objects = data.get_eval_batch(idx, args.split)
            score, _ = model(heads, rels, mode=mode, targets=None)
            all_ranks.append(ranker(
                score.float().cpu().numpy(), objects, data.filters_for_split(args.split),
                heads, rels, data.n_entities_for_split(args.split)
            ))
    ranks = np.concatenate(all_ranks) if all_ranks else np.empty(0)
    out = metrics(ranks)
    out.update({
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "selected_valid": ckpt.get("valid"),
        "mass_support_gate": cfg.get("mass_support_gate", "none"),
        "tie_policy": args.tie_policy,
    })
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.save_ranks:
        np.save(args.save_ranks, ranks)


if __name__ == "__main__":
    main()
