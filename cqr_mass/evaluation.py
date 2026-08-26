from __future__ import annotations

import numpy as np


def average_tie_ranks(scores, objects, filters, heads, relations, n_ent):
    """Filtered true average-tie ranks.

    rank = 1 + #(score > target) + 0.5 * (#ties - 1)
    after filtering other known positives for the same query.
    """
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise FloatingPointError("evaluation received non-finite entity scores")
    ranks = []
    for i in range(len(scores)):
        golds = np.flatnonzero(objects[i] > 0)
        blocked = np.zeros(n_ent, dtype=bool)
        known = filters[(int(heads[i]), int(relations[i]))]
        if known:
            blocked[np.fromiter(known, dtype=np.int64)] = True
        for gold in golds:
            eligible = ~blocked.copy()
            eligible[gold] = True
            target = scores[i, gold]
            vals = scores[i, eligible]
            rank = 1 + np.sum(vals > target) + 0.5 * max(np.sum(vals == target) - 1, 0)
            ranks.append(rank)
    return np.asarray(ranks, dtype=np.float64)


def ordinal_tie_ranks(scores, objects, filters, heads, relations, n_ent):
    """Filtered deterministic ordinal ranks.

    Scores are ordered descending. Equal scores are broken by ascending entity
    id, which is equivalent to a stable sort over the entity score vector.
    """
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise FloatingPointError("evaluation received non-finite entity scores")
    ranks = []
    entity_ids = np.arange(n_ent)
    for i in range(len(scores)):
        golds = np.flatnonzero(objects[i] > 0)
        blocked = np.zeros(n_ent, dtype=bool)
        known = filters[(int(heads[i]), int(relations[i]))]
        if known:
            blocked[np.fromiter(known, dtype=np.int64)] = True
        for gold in golds:
            eligible = ~blocked.copy()
            eligible[gold] = True
            target = scores[i, gold]
            ahead = (scores[i] > target) | (
                (scores[i] == target) & (entity_ids < gold)
            )
            rank = 1 + np.sum(ahead & eligible)
            ranks.append(rank)
    return np.asarray(ranks, dtype=np.float64)


def pessimistic_tie_ranks(scores, objects, filters, heads, relations, n_ent):
    """Filtered pessimistic ranks, placing the gold last within every tie."""
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise FloatingPointError("evaluation received non-finite entity scores")
    ranks = []
    for i in range(len(scores)):
        golds = np.flatnonzero(objects[i] > 0)
        blocked = np.zeros(n_ent, dtype=bool)
        known = filters[(int(heads[i]), int(relations[i]))]
        if known:
            blocked[np.fromiter(known, dtype=np.int64)] = True
        for gold in golds:
            eligible = ~blocked.copy()
            eligible[gold] = True
            target = scores[i, gold]
            vals = scores[i, eligible]
            rank = 1 + np.sum(vals > target) + max(np.sum(vals == target) - 1, 0)
            ranks.append(rank)
    return np.asarray(ranks, dtype=np.float64)


def metrics(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    if not len(r):
        return {"n": 0, "MRR": float("nan"), "H@1": float("nan"), "H@3": float("nan"), "H@10": float("nan")}
    return {
        "n": int(len(r)),
        "MRR": float(np.mean(1.0 / r)),
        "H@1": float(np.mean(r <= 1)),
        "H@3": float(np.mean(r <= 3)),
        "H@10": float(np.mean(r <= 10)),
    }
