from __future__ import annotations

import torch
import torch.nn as nn


class EdgeConditionedCQRLayer(nn.Module):
    """One learned message-passing layer on the packed quotient graph."""

    def __init__(self, dim: int = 24, rho_dim: int = 6):
        super().__init__()
        self.rho_proj = nn.Sequential(nn.Linear(rho_dim, dim, bias=False), nn.Tanh())
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(2 * dim, dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(4 * dim, dim), nn.Sigmoid())
        self.self_proj = nn.Linear(dim, dim, bias=False)
        self.update = nn.Sequential(nn.Linear(3 * dim, dim), nn.Tanh())
        self.norm = nn.LayerNorm(dim)
        for m in (self.rho_proj[0], self.q_proj, self.value, self.gate[0], self.self_proj, self.update[0]):
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.update[0].bias)

    def forward(self, z, src, dst, rho, q, node_graph, edge_graph, edge_active):
        n = z.size(0)
        if src.numel():
            rh = self.rho_proj(rho)
            qedge = self.q_proj(q[edge_graph])
            zs, zd = z[src], z[dst]
            gate = self.gate(torch.cat([zs, zd, rh, qedge], dim=-1))
            msg = gate * self.value(torch.cat([zs, rh + qedge], dim=-1))
            out_deg = torch.bincount(src, weights=edge_active, minlength=n).clamp_min(1.0)
            in_deg = torch.bincount(dst, weights=edge_active, minlength=n).clamp_min(1.0)
            msg = msg * torch.rsqrt(out_deg[src] * in_deg[dst]).unsqueeze(-1) * edge_active.unsqueeze(-1)
            agg = z.new_zeros((n, z.size(-1)))
            agg.index_add_(0, dst, msg)
        else:
            agg = z.new_zeros(z.shape)
        qnode = q[node_graph]
        delta = self.update(torch.cat([self.self_proj(z), agg, qnode], dim=-1))
        return self.norm(z + delta)


class PackedQuotientReasoner(nn.Module):
    """CQR component reasoner used by both global decoding and mass alignment."""

    def __init__(self, bank, n_rel: int, dim: int = 24, num_layer: int = 3):
        super().__init__()
        self.bank = bank
        self.n_rel = int(n_rel)
        self.dim = int(dim)
        self.node_init = nn.Linear(5, dim, bias=False)
        self.qrel = nn.Embedding(2 * n_rel, dim)
        self.head_seed = nn.Parameter(torch.empty(dim))
        self.init_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([EdgeConditionedCQRLayer(dim, 6) for _ in range(num_layer)])
        self.comp_score = nn.Sequential(nn.Linear(3 * dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))
        nn.init.xavier_uniform_(self.node_init.weight)
        nn.init.normal_(self.qrel.weight, std=0.05)
        nn.init.normal_(self.head_seed, std=0.05)
        nn.init.xavier_uniform_(self.comp_score[0].weight)
        nn.init.zeros_(self.comp_score[0].bias)
        nn.init.xavier_uniform_(self.comp_score[2].weight)

    @staticmethod
    def _segment_logsumexp(logits, groups, n_group):
        work = logits.float()
        maxv = work.new_full((n_group,), float("-inf"))
        maxv.scatter_reduce_(0, groups, work, reduce="amax", include_self=True)
        shifted = torch.exp(work - maxv[groups])
        sums = work.new_zeros((n_group,))
        sums.index_add_(0, groups, shifted)
        return maxv + torch.log(sums.clamp_min(1e-30))

    def forward(self, heads, rels, mode, device, targets=None):
        rels_t = torch.as_tensor(rels, dtype=torch.long, device=device)
        packed = self.bank.pack(heads, rels, mode, device, targets=targets)
        q = self.qrel(rels_t)
        z = self.node_init(packed.x) + q[packed.node_graph]
        z = z.clone()
        z[packed.head_index] = z[packed.head_index] + self.head_seed
        z = self.init_norm(torch.tanh(z))
        for layer in self.layers:
            z = layer(z, packed.src, packed.dst, packed.rho, q, packed.node_graph, packed.edge_graph, packed.edge_active)

        head_c = z[packed.head_index]
        ent_c = z[packed.entity_component_index]
        zh = head_c[packed.node_graph]
        qq = q[packed.node_graph]
        comp_logits = self.comp_score(torch.cat([z, zh, qq], dim=-1)).squeeze(-1)
        comp_entity_logits = comp_logits[packed.entity_component_index]

        comp_nll = None
        if packed.gold_component_index is not None:
            log_z = self._segment_logsumexp(comp_logits, packed.node_graph, packed.num_graphs)
            comp_nll = log_z - comp_logits[packed.gold_component_index]

        return (
            ent_c,
            head_c,
            q,
            comp_entity_logits,
            comp_nll,
            packed.entity_component_index,
        )
