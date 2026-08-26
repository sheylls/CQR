from __future__ import annotations

import torch
import torch.nn as nn


class CQRGlobalEntityDecoder(nn.Module):
    """Decode CQR component states plus entity structural roles to entity scores.

    This branch never reads A* hidden states. It is an independent global stream.
    """

    def __init__(self, cdim: int, edim: int, role_dim: int = 8):
        super().__init__()
        self.role = nn.Sequential(
            nn.Linear(role_dim, edim), nn.Tanh(), nn.Linear(edim, edim, bias=False)
        )
        self.comp = nn.Linear(cdim, edim, bias=False)
        self.head = nn.Linear(cdim, edim, bias=False)
        self.qc = nn.Linear(cdim, edim, bias=False)
        self.qa = nn.Linear(edim, edim, bias=False)
        self.mix = nn.Sequential(
            nn.Linear(5 * edim, 2 * edim),
            nn.GELU(),
            nn.Linear(2 * edim, edim),
            nn.LayerNorm(edim),
        )
        self.query = nn.Sequential(nn.Linear(cdim + edim, edim), nn.Tanh())
        self.score = nn.Sequential(
            nn.Linear(2 * edim, edim), nn.Tanh(), nn.Linear(edim, 1, bias=False)
        )
        for m in [
            self.role[0], self.role[2], self.comp, self.head, self.qc, self.qa,
            self.mix[0], self.mix[2], self.query[0], self.score[0], self.score[2],
        ]:
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ent_c, head_c, q_cqr, q_astar, role_feat):
        B, N, _ = ent_c.shape
        role = self.role(role_feat)
        c = self.comp(ent_c)
        h = self.head(head_c)[:, None, :].expand(B, N, -1)
        qc = self.qc(q_cqr)[:, None, :].expand(B, N, -1)
        qa = self.qa(q_astar)[:, None, :].expand(B, N, -1)
        gv = self.mix(torch.cat([c, h, qc, qa, role], dim=-1))
        gq = self.query(torch.cat([q_cqr, q_astar], dim=-1))[:, None, :].expand(B, N, -1)
        score = self.score(torch.cat([gv, gv * gq], dim=-1)).squeeze(-1)
        return score


class AdaptiveLateFusion(nn.Module):
    """Scale-stable late fusion of independent local and global score streams."""

    def __init__(self, cdim: int, edim: int, max_ratio: float = 0.35):
        super().__init__()
        self.max_ratio = float(max_ratio)
        self.gate = nn.Sequential(
            nn.Linear(2 * cdim + edim + 1, edim),
            nn.Tanh(),
            nn.Linear(edim, 1),
        )
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)
        self.last_ratio = torch.tensor(0.0)

    def forward(self, local_score, global_score, head_c, q_cqr, q_astar, fragmentation):
        gm = global_score.mean(dim=1, keepdim=True)
        gs = global_score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        gz = torch.tanh((global_score - gm) / gs)
        ls = local_score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)

        if fragmentation.ndim == 1:
            fragmentation = fragmentation.unsqueeze(-1)
        gate_in = torch.cat([head_c, q_cqr, q_astar, fragmentation], dim=-1)
        ratio = self.max_ratio * torch.sigmoid(self.gate(gate_in))
        self.last_ratio = ratio.mean().detach()
        return local_score + ratio * ls * gz
