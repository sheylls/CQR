from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.size(-1)))
    if index.numel():
        out.index_add_(0, index, src)
    return out


class GraphStore:
    """Static graph plus per-device GPU CSR cache for A* frontier expansion."""

    def __init__(self, loader):
        self.store = {}
        self._device = {}
        for mode, kg, n in (
            ("train_graph", np.asarray(loader.tra_KG), int(loader.n_ent)),
            ("inductive_graph", np.asarray(loader.ind_KG), int(loader.n_ent_ind)),
        ):
            real = kg[kg[:, 1] < 2 * loader.n_rel].astype(np.int64, copy=False)
            if len(real):
                order = np.argsort(real[:, 0], kind="stable").astype(np.int64)
                src, rel, dst = real[:, 0], real[:, 1], real[:, 2]
                deg = np.bincount(src, minlength=n).astype(np.int64)
                rowptr = np.zeros(n + 1, dtype=np.int64)
                rowptr[1:] = np.cumsum(deg)
            else:
                order = np.empty(0, np.int64)
                src = rel = dst = np.empty(0, np.int64)
                rowptr = np.zeros(n + 1, np.int64)
            self.store[mode] = {
                "src": src,
                "rel": rel,
                "dst": dst,
                "order": order,
                "rowptr": rowptr,
                "n": n,
                "E": len(real),
            }

    def tensors(self, mode: str, device):
        key = (mode, str(device))
        if key not in self._device:
            graph = self.store[mode]
            self._device[key] = {
                name: torch.as_tensor(graph[name], dtype=torch.long, device=device)
                for name in ("src", "rel", "dst", "order", "rowptr")
            }
        return self._device[key]

    @staticmethod
    def expand_csr(rowptr: torch.Tensor, order: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        if nodes.numel() == 0:
            return nodes.new_empty(0)
        starts = rowptr[nodes]
        degree = rowptr[nodes + 1] - starts
        total = int(degree.sum().item())
        if total == 0:
            return nodes.new_empty(0)
        prefix = torch.cumsum(degree, dim=0) - degree
        segment_start = torch.repeat_interleave(starts, degree)
        segment_prefix = torch.repeat_interleave(prefix, degree)
        within = torch.arange(total, device=nodes.device) - segment_prefix
        return order[segment_start + within]


class AStarConv(nn.Module):
    def __init__(self, dim: int, n_rel: int):
        super().__init__()
        self.relation = nn.Embedding(2 * n_rel, dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, dense_input, boundary, batch, src, rel, dst, batch_size: int, n_entity: int):
        if batch.numel():
            message = dense_input[batch, src] * self.relation(rel)
            aggregate = scatter_sum(
                message, batch * n_entity + dst, batch_size * n_entity
            ).view(batch_size, n_entity, -1)
        else:
            aggregate = dense_input.new_zeros(dense_input.shape)
        aggregate = aggregate + boundary
        return F.relu(self.norm(self.linear(torch.cat([dense_input, aggregate], dim=-1))))
