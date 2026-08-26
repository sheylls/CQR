from __future__ import annotations

import numpy as np
import torch


class QuotientEntityRoleBank:
    """Leakage-safe entity roles for decoding quotient states back to entities.

    The quotient decomposition removes the queried base relation and inverse,
    hence residual edges are internal to connected components.  Entity-specific
    roles are therefore defined by (a) residual centrality and (b) whether an
    entity is a gateway for the removed query relation across/same components.

    For training queries the current (h,r,t) edge is subtracted from the role
    counts before normalization, matching the exact target-edge deletion used by
    A* and packed CQR.  No target information is present at validation/test.
    """

    def __init__(self, loader, quotient_bank):
        self.loader = loader
        self.quotient_bank = quotient_bank
        self.n_rel = int(loader.n_rel)
        self.templates = {
            "train_graph": self._build_mode(np.asarray(loader.tra_KG), int(loader.n_ent), "train_graph"),
            "inductive_graph": self._build_mode(np.asarray(loader.ind_KG), int(loader.n_ent_ind), "inductive_graph"),
        }
        self._cache = {}

    def _build_mode(self, graph: np.ndarray, n_ent: int, mode: str):
        real = graph[graph[:, 1] < 2 * self.n_rel].astype(np.int64, copy=False)
        out = []
        for qrel in range(2 * self.n_rel):
            base = qrel % self.n_rel
            inv = base + self.n_rel
            labels = self.quotient_bank.templates[mode][qrel]["labels"]

            residual = real[(real[:, 1] != base) & (real[:, 1] != inv)]
            res_out = np.bincount(residual[:, 0], minlength=n_ent).astype(np.float32) if len(residual) else np.zeros(n_ent, np.float32)
            res_in = np.bincount(residual[:, 2], minlength=n_ent).astype(np.float32) if len(residual) else np.zeros(n_ent, np.float32)

            qedges = real[real[:, 1] == qrel]
            cross_out = np.zeros(n_ent, dtype=np.float32)
            cross_in = np.zeros(n_ent, dtype=np.float32)
            same_out = np.zeros(n_ent, dtype=np.float32)
            same_in = np.zeros(n_ent, dtype=np.float32)
            if len(qedges):
                src, dst = qedges[:, 0], qedges[:, 2]
                cross = labels[src] != labels[dst]
                if np.any(cross):
                    np.add.at(cross_out, src[cross], 1.0)
                    np.add.at(cross_in, dst[cross], 1.0)
                same = ~cross
                if np.any(same):
                    np.add.at(same_out, src[same], 1.0)
                    np.add.at(same_in, dst[same], 1.0)

            raw = np.stack([res_out, res_in, cross_out, cross_in, same_out, same_in], axis=-1).astype(np.float32)
            out.append({"raw": raw, "labels": labels.astype(np.int64, copy=False)})
        return out

    def _tensor_template(self, mode: str, qrel: int, device):
        key = (mode, int(qrel), str(device))
        if key not in self._cache:
            t = self.templates[mode][int(qrel)]
            self._cache[key] = {
                "raw": torch.as_tensor(t["raw"], dtype=torch.float32, device=device),
                "labels": torch.as_tensor(t["labels"], dtype=torch.long, device=device),
            }
        return self._cache[key]

    def _tensor_bank(self, mode: str, device):
        """Stack all relation templates once for vectorized batch lookup."""
        key = (mode, "__bank__", str(device))
        if key not in self._cache:
            self._cache[key] = {
                "raw": torch.stack([
                    torch.as_tensor(t["raw"], dtype=torch.float32, device=device)
                    for t in self.templates[mode]
                ], dim=0),
                "labels": torch.stack([
                    torch.as_tensor(t["labels"], dtype=torch.long, device=device)
                    for t in self.templates[mode]
                ], dim=0),
            }
        return self._cache[key]

    @staticmethod
    def _norm_col(x: torch.Tensor) -> torch.Tensor:
        peak = x.max().clamp_min(0.0)
        den = torch.log1p(peak).clamp_min(1.0)
        return torch.log1p(x.clamp_min(0.0)) / den

    def get(self, heads, rels, mode: str, device, targets=None):
        """Vectorized exact equivalent of the original per-query role builder."""
        heads_t = torch.as_tensor(np.asarray(heads, dtype=np.int64), dtype=torch.long, device=device)
        rels_t = torch.as_tensor(np.asarray(rels, dtype=np.int64), dtype=torch.long, device=device)
        if heads_t.numel() == 0:
            raise ValueError("empty role batch")

        bank = self._tensor_bank(mode, device)
        raw = bank["raw"].index_select(0, rels_t)
        labels = bank["labels"].index_select(0, rels_t)

        if targets is not None:
            raw = raw.clone()
            targets_t = torch.as_tensor(np.asarray(targets, dtype=np.int64), dtype=torch.long, device=device)
            b = torch.arange(heads_t.numel(), device=device)
            cross = labels[b, heads_t] != labels[b, targets_t]
            out_ch = torch.where(cross, torch.full_like(heads_t, 2), torch.full_like(heads_t, 4))
            in_ch = torch.where(cross, torch.full_like(heads_t, 3), torch.full_like(heads_t, 5))
            raw[b, heads_t, out_ch] -= 1.0
            raw[b, targets_t, in_ch] -= 1.0
            raw.clamp_min_(0.0)

        # Exact column-wise normalization, now batched over queries and all six columns.
        x = raw.clamp_min(0.0)
        peak = x.amax(dim=1, keepdim=True)
        den = torch.log1p(peak).clamp_min(1.0)
        norm = torch.log1p(x) / den
        feat = torch.cat([
            norm,
            (raw[:, :, 2:3] > 0).to(norm.dtype),
            (raw[:, :, 3:4] > 0).to(norm.dtype),
        ], dim=-1)
        return feat
