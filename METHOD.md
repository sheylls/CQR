# CQR: Method Overview

## 1. Inductive protocol

Each benchmark split contains a training graph (G1) and an inference graph (G2) with disjoint entity sets. Model parameters are learned from G1. Validation on G1 selects the checkpoint, while G2 is reserved for inductive evaluation.

For every training query `(h, r, t)`, the target edge and its inverse are removed from all structural evidence used by the model.

## 2. Local and component reasoning

The local branch applies query-conditioned A*-style propagation on the entity graph. A learned priority function selects the active frontier and produces entity-level scores.

For the component branch, the queried relation and its inverse are removed from the fact graph. Connected residual components are contracted into quotient nodes, and removed relation edges define directed connections between components. CQR performs learned message passing on this quotient graph and decodes the component states back to entity scores using structural roles.

The local and component branches remain independent until adaptive late score fusion.

## 3. Component-level score calibration

Let `s(v)` be the fused entity score, `g_c` the logit of component `c`, and

```text
m_c = logsumexp(s(v) for v in c).
```

The final score is

```text
s'(v) = s(v) + alpha * (g_C(v) - m_C(v)),
```

where `alpha` is a learned global scalar. Entities within the same component receive the same additive correction, so their relative ordering is preserved.

## 4. Objective and evaluation

The model is trained end to end from random initialization with entity, component, and global auxiliary objectives:

```text
L = CE(final_score, target)
  + lambda_component * component_NLL
  + lambda_global * CE(global_score, target).
```

Evaluation uses filtered true average-tie ranking. Checkpoint selection uses validation MRR; the test split is evaluated only after model selection.
