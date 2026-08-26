from __future__ import annotations

import torch
import torch.nn as nn

from .astar import AStarConv, GraphStore
from .entity_role import QuotientEntityRoleBank
from .global_branch import AdaptiveLateFusion, CQRGlobalEntityDecoder
from .mass_alignment import ComponentMassAlignment
from .quotient_reasoner import PackedQuotientReasoner
from .quotient_structure import QuotientStructureBank


class CQRMassAStar(nn.Module):
    """CQR model for inductive knowledge graph completion.

    A* local reasoning and CQR global decoding stay independent until late
    fusion. Component-level score alignment then adjusts component probabilities
    without changing entity order inside any component.
    """

    def __init__(
        self,
        loader,
        dim: int = 32,
        cqdim: int = 24,
        q_layers: int = 3,
        num_layer: int = 6,
        node_ratio: float = 0.05,
        test_node_ratio: float = 1.0,
        late_fusion_max_ratio: float = 0.35,
        mass_init_alpha: float = 0.20,
        mass_fixed_alpha: float | None = None,
        mass_support_gate: str = "none",
    ):
        super().__init__()
        self.loader = loader
        self.n_rel = int(loader.n_rel)
        self.dim = int(dim)
        self.num_layer = int(num_layer)
        self.node_ratio = float(node_ratio)
        self.test_node_ratio = float(test_node_ratio)
        self.degree_ratio = 1.0
        if mass_support_gate not in {
            "none",
            "binary",
            "one_sided",
            "q2_only",
            "q3_only",
            "directional",
            "evidence_consistent",
        }:
            raise ValueError(
                "mass_support_gate must be 'none', 'binary', 'one_sided', "
                "'q2_only', 'q3_only', 'directional', or 'evidence_consistent'"
            )
        self.mass_support_gate = mass_support_gate

        self.graph = GraphStore(loader)
        self.query = nn.Embedding(2 * self.n_rel, dim)
        self.cqr_bank = QuotientStructureBank(loader)
        self.cqr_reasoner = PackedQuotientReasoner(self.cqr_bank, self.n_rel, dim=cqdim, num_layer=q_layers)
        self.entity_role_bank = QuotientEntityRoleBank(loader, self.cqr_bank)

        self.layers = nn.ModuleList([AStarConv(dim, self.n_rel) for _ in range(num_layer)])
        self.heuristic_linear = nn.Linear(2 * dim, dim)
        self.heuristic_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, 1))
        self.global_decoder = CQRGlobalEntityDecoder(cqdim, dim, role_dim=8)
        self.late_fusion = AdaptiveLateFusion(cqdim, dim, max_ratio=late_fusion_max_ratio)
        self.mass_alignment = ComponentMassAlignment(
            init_alpha=mass_init_alpha,
            fixed_alpha=mass_fixed_alpha,
        )

        self.last_node_frac = 0.0
        self.last_edge_frac = 0.0
        self.last_global_fusion_ratio = torch.tensor(0.0)
        initial_mass_alpha = mass_init_alpha if mass_fixed_alpha is None else mass_fixed_alpha
        self.last_mass_alpha = torch.tensor(initial_mass_alpha)
        self.last_mass_gap = torch.tensor(0.0)
        self.last_global_score = None
        self.last_pre_mass_score = None

    @property
    def device(self):
        return next(self.parameters()).device

    def _priority(self, hidden, q_astar):
        B, N, _ = hidden.shape
        qa = q_astar[:, None, :].expand(B, N, -1)
        heur = self.heuristic_linear(torch.cat([hidden, qa], dim=-1))
        return self.heuristic_mlp(hidden * heur).squeeze(-1)

    def _remove_easy(self, bt, ss, rr, dd, heads, rels, targets):
        if targets is None or bt.numel() == 0:
            return bt, ss, rr, dd
        dev = bt.device
        h = torch.as_tensor(heads, dtype=torch.long, device=dev)[bt]
        r = torch.as_tensor(rels, dtype=torch.long, device=dev)[bt]
        t = torch.as_tensor(targets, dtype=torch.long, device=dev)[bt]
        inv = torch.where(r < self.n_rel, r + self.n_rel, r - self.n_rel)
        keep = ~(((ss == h) & (rr == r) & (dd == t)) | ((ss == t) & (rr == inv) & (dd == h)))
        return bt[keep], ss[keep], rr[keep], dd[keep]

    def _select(self, score, known, mode, heads=None, rels=None, targets=None):
        meta = self.graph.store[mode]
        g = self.graph.tensors(mode, self.device)
        B, N = score.shape
        ratio = self.node_ratio if self.training else self.test_node_ratio

        if ratio >= 1.0 and self.degree_ratio >= 1.0:
            src_all, rel_all, dst_all = g["src"], g["rel"], g["dst"]
            active = known[:, src_all]
            bt, ei = torch.nonzero(active, as_tuple=True)
            if ei.numel():
                ss, rr, dd = src_all[ei], rel_all[ei], dst_all[ei]
            else:
                bt = ss = rr = dd = torch.empty(0, dtype=torch.long, device=self.device)
            self.last_node_frac = float(known.sum().item()) / max(B * N, 1)
            self.last_edge_frac = float(ei.numel()) / max(B * meta["E"], 1)
            return self._remove_easy(bt, ss, rr, dd, heads, rels, targets)

        k_budget = max(int(ratio * N), 1)
        e_budget = max(int(self.degree_ratio * ratio * meta["E"]), 1)
        batches, srcs, rels_out, dsts = [], [], [], []
        nsrc = nedge = 0
        for b in range(B):
            ids = torch.nonzero(known[b], as_tuple=False).squeeze(1)
            if ids.numel() == 0:
                continue
            k = min(k_budget, ids.numel())
            top = ids[torch.topk(score[b, ids], k=k, largest=True, sorted=False).indices]
            nsrc += int(top.numel())
            cand = self.graph.expand_csr(g["rowptr"], g["order"], top)
            if cand.numel() == 0:
                continue
            e = min(e_budget, cand.numel())
            if e < cand.numel():
                dst_cand = g["dst"][cand]
                cand = cand[torch.topk(score[b, dst_cand], k=e, largest=True, sorted=False).indices]
            nedge += int(cand.numel())
            batches.append(torch.full((cand.numel(),), b, dtype=torch.long, device=self.device))
            srcs.append(g["src"][cand])
            rels_out.append(g["rel"][cand])
            dsts.append(g["dst"][cand])

        if batches:
            bt, ss, rr, dd = map(torch.cat, (batches, srcs, rels_out, dsts))
        else:
            bt = ss = rr = dd = torch.empty(0, dtype=torch.long, device=self.device)
        self.last_node_frac = nsrc / max(B * N, 1)
        self.last_edge_frac = nedge / max(B * meta["E"], 1)
        return self._remove_easy(bt, ss, rr, dd, heads, rels, targets)

    def forward(self, heads, rels, mode="train_graph", targets=None):
        dev = self.device
        B = len(heads)
        N = self.loader.n_ent if mode == "train_graph" else self.loader.n_ent_ind
        ht = torch.as_tensor(heads, dtype=torch.long, device=dev)
        rt = torch.as_tensor(rels, dtype=torch.long, device=dev)
        q_astar = self.query(rt)

        ent_c, head_c, q_cqr, comp_entity_logits, comp_nll, ent_comp_id = self.cqr_reasoner(
            heads, rels, mode, dev, targets=targets
        )
        role_feat = self.entity_role_bank.get(heads, rels, mode, dev, targets=targets)
        global_score = self.global_decoder(ent_c, head_c, q_cqr, q_astar, role_feat)

        boundary = torch.zeros((B, N, self.dim), device=dev)
        boundary[torch.arange(B, device=dev), ht] = q_astar
        hidden = boundary.clone()
        known = torch.zeros((B, N), dtype=torch.bool, device=dev)
        known[torch.arange(B, device=dev), ht] = True
        score = self._priority(hidden, q_astar)

        for layer in self.layers:
            bt, ss, rr, dd = self._select(score, known, mode, heads=heads, rels=rels, targets=targets)
            layer_input = torch.sigmoid(score).unsqueeze(-1) * hidden
            out = layer(layer_input, boundary, bt, ss, rr, dd, B, N)
            mask = torch.zeros((B, N), dtype=torch.bool, device=dev)
            if bt.numel():
                mask[bt, dd] = True
            mask[torch.arange(B, device=dev), ht] = True
            hidden = torch.where(mask.unsqueeze(-1), hidden + out, hidden)
            known |= mask
            newscore = self._priority(hidden, q_astar)
            score = torch.where(mask, newscore, score)

        n_comp = (ent_comp_id.max(dim=1).values - ent_comp_id.min(dim=1).values + 1).to(hidden.dtype)
        fragmentation = n_comp / float(N)
        fused_score = self.late_fusion(score, global_score, head_c, q_cqr, q_astar, fragmentation)
        support_mask = known if self.mass_support_gate != "none" else None
        final_score = self.mass_alignment(
            fused_score,
            comp_entity_logits,
            ent_comp_id,
            support_mask=support_mask,
            support_gate_mode=self.mass_support_gate,
        )

        self.last_global_fusion_ratio = self.late_fusion.last_ratio
        self.last_mass_alpha = self.mass_alignment.last_alpha
        self.last_mass_gap = self.mass_alignment.last_gap
        self.last_global_score = global_score
        self.last_pre_mass_score = fused_score
        return final_score, comp_nll
