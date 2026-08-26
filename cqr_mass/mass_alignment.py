from __future__ import annotations

import math
import torch
import torch.nn as nn


class ComponentMassAlignment(nn.Module):
    """Align fine entity mass with CQR component logits.

    For entity v in component C(v):
        s'_v = s_v + alpha * (g_{C(v)} - LSE_{u in C(v)} s_u)

    The correction is constant inside each component, therefore intra-component
    entity ranking is exactly preserved.
    """

    def __init__(self, init_alpha: float = 0.20, fixed_alpha: float | None = None):
        super().__init__()
        if not 0.0 < init_alpha < 1.0:
            raise ValueError("init_alpha must be in (0, 1)")
        if fixed_alpha is not None and not 0.0 <= fixed_alpha <= 1.0:
            raise ValueError("fixed_alpha must be in [0, 1]")
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(init_alpha / (1.0 - init_alpha))))
        self.fixed_alpha = None if fixed_alpha is None else float(fixed_alpha)
        initial_alpha = init_alpha if fixed_alpha is None else fixed_alpha
        self.last_alpha = torch.tensor(initial_alpha)
        self.last_gap = torch.tensor(0.0)

    @staticmethod
    def segment_logsumexp(values: torch.Tensor, groups: torch.Tensor, n_group: int) -> torch.Tensor:
        work = values.float()
        maxv = work.new_full((n_group,), float("-inf"))
        maxv.scatter_reduce_(0, groups, work, reduce="amax", include_self=True)
        shifted = torch.exp(work - maxv[groups])
        sums = work.new_zeros((n_group,))
        sums.index_add_(0, groups, shifted)
        return maxv + torch.log(sums.clamp_min(1e-30))

    def forward(
        self,
        entity_score: torch.Tensor,
        component_logit_per_entity: torch.Tensor,
        entity_component_index: torch.Tensor,
        support_mask: torch.Tensor | None = None,
        support_gate_mode: str | None = None,
    ) -> torch.Tensor:
        if entity_score.shape != component_logit_per_entity.shape:
            raise ValueError("entity_score and component_logit_per_entity must have identical shape")
        if entity_component_index.shape != entity_score.shape:
            raise ValueError("entity_component_index must have shape [B, N]")
        if support_mask is not None and support_mask.shape != entity_score.shape:
            raise ValueError("support_mask must have shape [B, N]")
        if support_gate_mode is None:
            support_gate_mode = "binary" if support_mask is not None else "none"
        if support_gate_mode not in {
            "none",
            "binary",
            "one_sided",
            "q2_only",
            "q3_only",
            "directional",
            "evidence_consistent",
        }:
            raise ValueError(
                "support_gate_mode must be 'none', 'binary', 'one_sided', "
                "'q2_only', 'q3_only', 'directional', or 'evidence_consistent'"
            )
        if support_gate_mode != "none" and support_mask is None:
            raise ValueError("support_mask is required when support_gate_mode is enabled")

        if self.fixed_alpha == 0.0:
            self.last_alpha = entity_score.new_tensor(0.0)
            self.last_gap = entity_score.new_tensor(0.0)
            return entity_score

        groups = entity_component_index.reshape(-1)
        values = entity_score.reshape(-1)
        n_group = int(groups.max().item()) + 1
        local_mass = self.segment_logsumexp(values, groups, n_group)
        local_mass_per_entity = local_mass[groups].view_as(entity_score)
        gap = component_logit_per_entity - local_mass_per_entity
        if support_gate_mode != "none":
            component_support = local_mass.new_zeros((n_group,))
            component_support.scatter_reduce_(
                0,
                groups,
                support_mask.reshape(-1).to(local_mass.dtype),
                reduce="amax",
                include_self=True,
            )
            supported = component_support[groups].view_as(entity_score) > 0
            if support_gate_mode == "binary":
                gap = gap * supported.to(gap.dtype)
            elif support_gate_mode in {"one_sided", "q3_only"}:
                gap = torch.where(supported, gap, gap.clamp_max(0.0))
            elif support_gate_mode == "q2_only":
                gap = torch.where(supported, gap.clamp_min(0.0), gap)
            else:  # directional and evidence_consistent are equivalent.
                gap = torch.where(supported, gap.clamp_min(0.0), gap.clamp_max(0.0))
        alpha = (
            torch.sigmoid(self.alpha_logit)
            if self.fixed_alpha is None
            else entity_score.new_tensor(self.fixed_alpha)
        )
        self.last_alpha = alpha.detach()
        self.last_gap = gap.abs().mean().detach()
        return entity_score + alpha * gap
