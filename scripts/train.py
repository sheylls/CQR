#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cqr_mass.config import load_config
from cqr_mass.data import load_dataset
from cqr_mass.evaluation import average_tie_ranks, metrics
from cqr_mass.model import CQRMassAStar


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_effective_batches(order, batch_size, batch_per_epoch=None):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_per_epoch is not None:
        batch_per_epoch = int(batch_per_epoch)
        if batch_per_epoch <= 0:
            raise ValueError("batch_per_epoch must be positive when provided")
        limit = min(len(order), batch_size * batch_per_epoch)
    else:
        limit = len(order)
    for start in range(0, limit, batch_size):
        yield order[start : min(start + batch_size, limit)]


def iter_micro_batches(effective_indices, micro_batch_size):
    micro_batch_size = int(micro_batch_size)
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    total = len(effective_indices)
    if total == 0:
        return
    for start in range(0, total, micro_batch_size):
        index = effective_indices[start : start + micro_batch_size]
        yield index, len(index) / total


def build_model(data, cfg):
    return CQRMassAStar(
        data,
        dim=cfg["entity_dim"],
        cqdim=cfg["cqr_dim"],
        q_layers=cfg["cqr_layers"],
        num_layer=cfg["astar_layers"],
        node_ratio=cfg["train_node_ratio"],
        test_node_ratio=cfg["test_node_ratio"],
        late_fusion_max_ratio=cfg["late_fusion_max_ratio"],
        mass_init_alpha=cfg["mass_init_alpha"],
        mass_fixed_alpha=cfg.get("mass_fixed_alpha"),
        mass_support_gate=cfg.get("mass_support_gate", "none"),
    )


@torch.inference_mode()
def evaluate(model, data, split, batch_size):
    model.eval()
    n = data.n_valid if split == "valid" else data.n_test
    mode = data.mode_for_split(split)
    ranks = []
    for st in range(0, n, batch_size):
        idx = np.arange(st, min(st + batch_size, n))
        heads, rels, objects = data.get_eval_batch(idx, split)
        score, _ = model(heads, rels, mode=mode, targets=None)
        ranks.append(
            average_tie_ranks(
                score.float().cpu().numpy(), objects,
                data.filters_for_split(split), heads, rels,
                data.n_entities_for_split(split),
            )
        )
    return metrics(np.concatenate(ranks) if ranks else np.empty(0))


def main():
    p = argparse.ArgumentParser(description="Train CQR for inductive knowledge graph completion")
    p.add_argument("--dataset", required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--micro-batch-size", type=int)
    p.add_argument("--batch-per-epoch", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--astar-layers", type=int)
    p.add_argument("--late-fusion-max-ratio", type=float)
    p.add_argument("--mass-init-alpha", type=float)
    p.add_argument("--mass-fixed-alpha", type=float)
    p.add_argument("--component-loss-weight", type=float)
    p.add_argument("--global-aux-weight", type=float)
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
    p.add_argument("--run-dir")
    p.add_argument("--amp", action="store_true", help="enable CUDA autocast/GradScaler")
    p.add_argument("--num-threads", type=int, default=4)
    args = p.parse_args()

    if not args.device.startswith("cuda"):
        raise SystemExit("This clean runner is the GPU version. Use --device cuda:0 (or another CUDA device).")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")

    torch.set_num_threads(max(args.num_threads, 1))
    cfg = load_config(ROOT, args.dataset)
    for k, v in (
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("micro_batch_size", args.micro_batch_size),
        ("batch_per_epoch", args.batch_per_epoch),
        ("lr", args.lr),
        ("astar_layers", args.astar_layers),
        ("late_fusion_max_ratio", args.late_fusion_max_ratio),
        ("mass_init_alpha", args.mass_init_alpha),
        ("mass_fixed_alpha", args.mass_fixed_alpha),
        ("component_loss_weight", args.component_loss_weight),
        ("global_aux_weight", args.global_aux_weight),
        ("mass_support_gate", args.mass_support_gate),
    ):
        if v is not None:
            cfg[k] = v
    seed_all(args.seed)

    effective_batch_size = int(cfg["batch_size"])
    micro_batch_size = int(cfg.get("micro_batch_size", effective_batch_size))
    if effective_batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive")
    if micro_batch_size > effective_batch_size:
        raise ValueError("micro_batch_size cannot exceed batch_size")
    batch_per_epoch = cfg.get("batch_per_epoch")
    if batch_per_epoch is not None and int(batch_per_epoch) <= 0:
        raise ValueError("batch_per_epoch must be positive when provided")
    if not 0.0 <= float(cfg["late_fusion_max_ratio"]) <= 1.0:
        raise ValueError("late_fusion_max_ratio must be in [0, 1]")
    if not 0.0 < float(cfg["mass_init_alpha"]) < 1.0:
        raise ValueError("mass_init_alpha must be in (0, 1)")
    name = f"{args.dataset}_{args.version}"
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    def run_is_complete():
        history_path = run_dir / "history.json"
        if not history_path.exists() or not (run_dir / "best.pt").exists():
            return False
        try:
            return len(json.loads(history_path.read_text(encoding="utf-8"))) >= int(cfg["epochs"])
        except (json.JSONDecodeError, OSError, TypeError):
            return False

    if run_is_complete():
        print(f"skip completed run: {run_dir}", flush=True)
        return

    lock_path = run_dir / ".training.lock"
    lock_fd = None
    wait_started = time.monotonic()
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            if run_is_complete():
                print(f"skip completed run: {run_dir}", flush=True)
                return
            if time.monotonic() - wait_started > 6 * 60 * 60:
                raise TimeoutError(f"timed out waiting for training lock: {lock_path}")
            print(f"waiting for active run: {run_dir}", flush=True)
            time.sleep(10)

    def release_training_lock():
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(release_training_lock)

    data = load_dataset(ROOT, args.dataset, args.version, micro_batch_size)
    model = build_model(data, cfg).to(torch.device(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    # BF16 keeps the dynamic range needed by dense entity logits on modern GPUs.
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    (run_dir / "config.json").write_text(json.dumps({**cfg, "dataset": args.dataset, "version": args.version, "seed": args.seed}, indent=2), encoding="utf-8")

    best_mrr = -1.0
    order_rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = order_rng.permutation(data.n_train)
        epoch_size = min(
            data.n_train,
            effective_batch_size * int(batch_per_epoch),
        ) if batch_per_epoch is not None else data.n_train
        sums = {"loss": 0.0, "entity": 0.0, "component": 0.0, "global": 0.0}
        seen = 0
        optimizer_steps = 0
        for effective_idx in iter_effective_batches(
            order,
            effective_batch_size,
            batch_per_epoch=batch_per_epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            for idx, loss_weight in iter_micro_batches(effective_idx, micro_batch_size):
                triples = data.get_train_batch(idx)
                heads, rels, targets = triples[:, 0], triples[:, 1], triples[:, 2]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                    score, comp_nll = model(heads, rels, mode="train_graph", targets=targets)
                    target_t = torch.as_tensor(targets, dtype=torch.long, device=model.device)
                    entity_ce = F.cross_entropy(score, target_t)
                    comp_loss = comp_nll.mean() if comp_nll is not None else score.new_zeros(())
                    global_ce = F.cross_entropy(model.last_global_score, target_t)
                    loss = (
                        entity_ce
                        + cfg["component_loss_weight"] * comp_loss
                        + cfg["global_aux_weight"] * global_ce
                    )
                if not torch.isfinite(score).all():
                    raise FloatingPointError(
                        f"non-finite entity scores at epoch {epoch}, sample {seen}"
                    )
                if comp_nll is not None and not torch.isfinite(comp_nll).all():
                    raise FloatingPointError(
                        f"non-finite component loss at epoch {epoch}, sample {seen}"
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch}, sample {seen}"
                    )
                scaler.scale(loss * loss_weight).backward()

                b = len(idx)
                seen += b
                sums["loss"] += float(loss.detach()) * b
                sums["entity"] += float(entity_ce.detach()) * b
                sums["component"] += float(comp_loss.detach()) * b
                sums["global"] += float(global_ce.detach()) * b

            if cfg.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            if optimizer_steps % cfg.get("log_every", 50) == 0:
                print(
                    f"epoch={epoch} train={seen}/{epoch_size} "
                    f"loss={sums['loss']/seen:.5f} alpha={float(model.last_mass_alpha):.4f} "
                    f"fusion={float(model.last_global_fusion_ratio):.4f}",
                    flush=True,
                )

        if any(
            p.requires_grad and not torch.isfinite(p).all()
            for p in model.parameters()
        ):
            raise FloatingPointError(f"non-finite model parameter after epoch {epoch}")
        valid = evaluate(model, data, "valid", cfg.get("eval_batch_size", cfg["batch_size"]))
        row = {
            "epoch": epoch,
            "train": {k: v / max(seen, 1) for k, v in sums.items()},
            "valid": valid,
            "mass_alpha": float(model.last_mass_alpha),
            "fusion_ratio": float(model.last_global_fusion_ratio),
            "schedule": {
                "samples_seen": seen,
                "samples_available": data.n_train,
                "optimizer_steps": optimizer_steps,
                "effective_batch_size": effective_batch_size,
                "micro_batch_size": micro_batch_size,
                "batch_per_epoch": batch_per_epoch,
            },
        }
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)

        ckpt = {
            "model": model.state_dict(),
            "config": cfg,
            "dataset": args.dataset,
            "version": args.version,
            "seed": args.seed,
            "epoch": epoch,
            "valid": valid,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if valid["MRR"] > best_mrr:
            best_mrr = valid["MRR"]
            torch.save(ckpt, run_dir / "best.pt")
            print(f"saved best.pt: epoch={epoch}, valid_MRR={best_mrr:.6f}", flush=True)

    print(f"done. best valid MRR={best_mrr:.6f}; test was NOT touched by training.")


if __name__ == "__main__":
    main()
