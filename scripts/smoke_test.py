#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cqr_mass.data import AStarNetInductiveData
from cqr_mass.mass_alignment import ComponentMassAlignment
from cqr_mass.model import CQRMassAStar
from cqr_mass.quotient_reasoner import PackedQuotientReasoner
from cqr_mass.evaluation import average_tie_ranks


def data_test():
    for ds in ("FB15k-237", "WN18RR", "NELL-995"):
        for v in ("v1", "v2", "v3", "v4"):
            d = AStarNetInductiveData(ROOT, ds, v, batch_size=2)
            assert d.n_ent > 0 and d.n_ent_ind > 0 and d.n_rel > 0
            assert d.n_train > 0 and d.n_valid > 0 and d.n_test > 0
            assert d.tra_KG.shape[1] == 3 and d.ind_KG.shape[1] == 3
            print(f"OK data {ds}/{v}: G1_ent={d.n_ent} G2_ent={d.n_ent_ind} rel={d.n_rel} trainQ={d.n_train} validQ={d.n_valid} testQ={d.n_test}")

def mass_invariant_test():
    torch.manual_seed(0)
    score = torch.randn(2, 5)
    comp = torch.tensor([[0, 0, 1, 1, 1], [2, 2, 3, 4, 4]])
    clogit = torch.tensor([[0.3, 0.3, -0.1, -0.1, -0.1], [0.2, 0.2, 1.0, -0.5, -0.5]])
    layer = ComponentMassAlignment(0.2)
    out = layer(score, clogit, comp)
    for b in range(score.size(0)):
        for c in torch.unique(comp[b]):
            ids = torch.where(comp[b] == c)[0]
            if ids.numel() > 1:
                before = score[b, ids][:, None] - score[b, ids][None, :]
                after = out[b, ids][:, None] - out[b, ids][None, :]
                assert torch.allclose(before, after, atol=1e-6)
    print("OK mass invariant: intra-component score differences exactly preserved")


def amp_segment_logsumexp_test():
    if not torch.cuda.is_available():
        print("SKIP AMP segment logsumexp: CUDA is unavailable")
        return

    logits = torch.tensor([1.0, 2.0, -3.0, 4.0], device="cuda", dtype=torch.float16)
    groups = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.long)
    expected = torch.stack((
        torch.logsumexp(logits[:2].float(), dim=0),
        torch.logsumexp(logits[2:].float(), dim=0),
    ))
    implementations = (
        PackedQuotientReasoner._segment_logsumexp,
        ComponentMassAlignment.segment_logsumexp,
    )
    with torch.amp.autocast("cuda", enabled=True):
        for implementation in implementations:
            actual = implementation(logits, groups, 2)
            torch.testing.assert_close(actual.float(), expected)
    print("OK AMP segment logsumexp: FP16 inputs accumulate safely in FP32")


def nonfinite_evaluation_rejected_test():
    scores = np.asarray([[np.nan, 0.0]], dtype=np.float32)
    objects = np.asarray([[1.0, 0.0]], dtype=np.float32)
    filters = {(0, 0): set()}
    try:
        average_tie_ranks(scores, objects, filters, np.asarray([0]), np.asarray([0]), 2)
    except FloatingPointError:
        print("OK evaluation rejects non-finite scores")
        return
    raise AssertionError("evaluation accepted non-finite scores")


def model_forward_test():
    # Small CPU smoke only. Production runner is GPU-only.
    d = AStarNetInductiveData(ROOT, "FB15k-237", "v1", batch_size=1)
    m = CQRMassAStar(d, dim=8, cqdim=8, q_layers=1, num_layer=1, node_ratio=0.02, test_node_ratio=0.02)
    tri = d.get_train_batch([0])
    score, comp = m(tri[:, 0], tri[:, 1], mode="train_graph", targets=tri[:, 2])
    assert score.shape == (1, d.n_ent)
    assert comp is not None and comp.shape == (1,)
    assert torch.isfinite(score).all() and torch.isfinite(comp).all()
    print("OK model forward/backbone shape")


if __name__ == "__main__":
    mass_invariant_test()
    amp_segment_logsumexp_test()
    nonfinite_evaluation_rejected_test()
    data_test()
    model_forward_test()
