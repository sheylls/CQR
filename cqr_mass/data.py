from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _index(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    out = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
            out[" ".join(parts[:-1])] = int(parts[-1])
        else:
            out[line] = i
    return out


def _read_triples(path: Path, e2id, r2id):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        h, r, t = line.split("\t") if "\t" in line else line.split()
        rows.append([e2id[h], r2id[r], e2id[t]])
    return rows


def _double(rows: Iterable[Sequence[int]], n_rel: int):
    out = []
    for h, r, t in rows:
        h, r, t = int(h), int(r), int(t)
        out.append([h, r, t])
        out.append([t, r + n_rel, h])
    return out


def _graph(rows, n_ent: int, n_rel: int):
    directed = np.asarray(_double(rows, n_rel), dtype=np.int64)
    if directed.size == 0:
        directed = np.empty((0, 3), dtype=np.int64)
    ident = np.stack(
        [np.arange(n_ent), np.full(n_ent, 2 * n_rel), np.arange(n_ent)], axis=1
    ).astype(np.int64)
    return np.concatenate([directed, ident], axis=0)


def _query_groups(rows, n_rel: int):
    grouped = defaultdict(list)
    for h, r, t in _double(rows, n_rel):
        grouped[(int(h), int(r))].append(int(t))
    keys = sorted(grouped)
    return keys, [np.asarray(grouped[k], dtype=np.int64) for k in keys]


def _filters(rows_list, n_rel: int):
    grouped = defaultdict(set)
    for rows in rows_list:
        for h, r, t in _double(rows, n_rel):
            grouped[(int(h), int(r))].add(int(t))
    return grouped


class _BaseData:
    batch_size: int

    def get_train_batch(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return self.train_triples[idx]

    def get_eval_batch(self, idx, split="valid"):
        idx = np.asarray(idx, dtype=np.int64)
        if split == "valid":
            q, a, n_ent = self.valid_q, self.valid_a, self.n_ent
        elif split == "test":
            q, a, n_ent = self.test_q, self.test_a, self.n_ent_ind
        else:
            raise ValueError(f"unknown split: {split}")
        qa = np.asarray(q, dtype=np.int64)
        heads, rels = qa[idx, 0], qa[idx, 1]
        objects = np.zeros((len(idx), n_ent), dtype=np.float32)
        for i, j in enumerate(idx):
            objects[i, a[int(j)]] = 1.0
        return heads, rels, objects

    def mode_for_split(self, split: str):
        return "train_graph" if split == "valid" else "inductive_graph"

    def filters_for_split(self, split: str):
        return self.val_filters if split == "valid" else self.tst_filters

    def n_entities_for_split(self, split: str):
        return self.n_ent if split == "valid" else self.n_ent_ind


class AStarNetInductiveData(_BaseData):
    """A*Net-style G1/G2 inductive protocol, full-entity objective.

    Train graph  : G1/train
    Train queries: G1/train + inverse queries
    Valid graph  : G1/train, queries G1/valid
    Test graph   : G2/train, queries G2/valid + G2/test

    Other known positives are filtered only at ranking time. During training the
    current target edge and its inverse are deleted inside the model.
    """

    def __init__(self, root, dataset="WN18RR", version="v1", batch_size=32):
        root = Path(root)
        base = root / "datasets" / "inductive" / dataset / version
        g1, g2 = base / "G1", base / "G2"
        if not g1.is_dir() or not g2.is_dir():
            raise FileNotFoundError(
                f"inductive split not found: {base}. Expected G1/ and G2/."
            )
        self.benchmark = "inductive"
        self.dataset = dataset
        self.version = version
        self.batch_size = int(batch_size)

        self.entity2id = _index(g1 / "entities.txt")
        self.entity2id_ind = _index(g2 / "entities.txt")
        self.relation2id = _index(g1 / "relations.txt")
        self.n_ent = len(self.entity2id)
        self.n_ent_ind = len(self.entity2id_ind)
        self.n_rel = len(self.relation2id)

        self.tra_train = _read_triples(g1 / "train.txt", self.entity2id, self.relation2id)
        self.tra_valid = _read_triples(g1 / "valid.txt", self.entity2id, self.relation2id)
        self.tra_test_filter = _read_triples(g1 / "test.txt", self.entity2id, self.relation2id)

        self.ind_train = _read_triples(g2 / "train.txt", self.entity2id_ind, self.relation2id)
        self.ind_valid = _read_triples(g2 / "valid.txt", self.entity2id_ind, self.relation2id)
        self.ind_test = _read_triples(g2 / "test.txt", self.entity2id_ind, self.relation2id)

        self.tra_KG = _graph(self.tra_train, self.n_ent, self.n_rel)
        self.ind_KG = _graph(self.ind_train, self.n_ent_ind, self.n_rel)
        self.train_triples = np.asarray(_double(self.tra_train, self.n_rel), dtype=np.int64)
        self.n_train = int(len(self.train_triples))

        self.valid_q, self.valid_a = _query_groups(self.tra_valid, self.n_rel)
        q1, a1 = _query_groups(self.ind_valid, self.n_rel)
        q2, a2 = _query_groups(self.ind_test, self.n_rel)
        self.test_q, self.test_a = q1 + q2, a1 + a2
        self.n_valid, self.n_test = len(self.valid_q), len(self.test_q)

        self.val_filters = _filters(
            [self.tra_train, self.tra_valid, self.tra_test_filter], self.n_rel
        )
        self.tst_filters = _filters(
            [self.ind_train, self.ind_valid, self.ind_test], self.n_rel
        )


def load_dataset(root, dataset: str, version="v1", batch_size=32):
    if dataset not in {"FB15k-237", "WN18RR", "NELL-995"}:
        raise ValueError("dataset must be one of FB15k-237, WN18RR, NELL-995")
    if version not in {"v1", "v2", "v3", "v4"}:
        raise ValueError("version must be v1, v2, v3 or v4")
    return AStarNetInductiveData(root, dataset, version, batch_size)
