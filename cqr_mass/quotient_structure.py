from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass
class PackedQuotientBatch:
    x: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    rho: torch.Tensor
    edge_active: torch.Tensor
    node_graph: torch.Tensor
    edge_graph: torch.Tensor
    head_index: torch.Tensor
    entity_component_index: torch.Tensor
    gold_component_index: torch.Tensor | None
    num_graphs: int
    num_entities: int


class QuotientStructureBank:
    """Static residual decomposition + GPU query deltas.

    The expensive connected-component decomposition is built once per base
    relation and graph split. Query-specific training deletion never rebuilds a
    SciPy graph. For a training query (h, r, t), the only structural change is

        Q_r[C(h), C(t)] <- Q_r[C(h), C(t)] - 1.

    That delta is applied to packed GPU tensors for the whole mini-batch.
    """

    def __init__(self, loader):
        self.loader = loader
        self.n_rel = int(loader.n_rel)
        self.structures = {
            "train_graph": self._build_mode(np.asarray(loader.tra_KG), int(loader.n_ent)),
            "inductive_graph": self._build_mode(np.asarray(loader.ind_KG), int(loader.n_ent_ind)),
        }
        self.templates = {
            mode: [self._oriented_template(mode, qrel) for qrel in range(2 * self.n_rel)]
            for mode in ("train_graph", "inductive_graph")
        }
        self._device_cache = {}

    def _build_mode(self, graph: np.ndarray, n_ent: int):
        real = graph[graph[:, 1] < 2 * self.n_rel].astype(np.int64, copy=False)
        out = []
        for base_r in range(self.n_rel):
            inv_r = base_r + self.n_rel
            residual = real[(real[:, 1] != base_r) & (real[:, 1] != inv_r)]
            if len(residual):
                A = sparse.csr_matrix(
                    (np.ones(len(residual), dtype=np.float32), (residual[:, 0], residual[:, 2])),
                    shape=(n_ent, n_ent),
                    dtype=np.float32,
                )
                A = (A + A.T).tocsr()
                if A.nnz:
                    A.data[:] = 1.0
            else:
                A = sparse.csr_matrix((n_ent, n_ent), dtype=np.float32)
            n_comp, labels = connected_components(A, directed=False)
            labels = labels.astype(np.int64, copy=False)
            sizes = np.bincount(labels, minlength=n_comp).astype(np.float32)

            qedges = real[real[:, 1] == base_r]
            Q = sparse.csr_matrix((n_comp, n_comp), dtype=np.float32)
            if len(qedges):
                ch, ct = labels[qedges[:, 0]], labels[qedges[:, 2]]
                cross = ch != ct
                if np.any(cross):
                    pairs = np.stack([ch[cross], ct[cross]], axis=1)
                    uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
                    Q = sparse.csr_matrix(
                        (cnt.astype(np.float32), (uniq[:, 0], uniq[:, 1])),
                        shape=(n_comp, n_comp),
                        dtype=np.float32,
                    )
            out.append(
                {
                    "labels": labels,
                    "sizes": sizes,
                    "Q": Q,
                    "n_comp": int(n_comp),
                    "n_ent": int(n_ent),
                }
            )
        return out

    def _oriented_template(self, mode: str, qrel: int):
        st = self.structures[mode][int(qrel) % self.n_rel]
        Qbase = st["Q"]
        Q = Qbase if int(qrel) < self.n_rel else Qbase.T.tocsr()
        n = int(st["n_comp"])
        sizes = st["sizes"].astype(np.float32, copy=False)

        out_mass = np.asarray(Q.sum(axis=1)).ravel().astype(np.float32)
        in_mass = np.asarray(Q.sum(axis=0)).ravel().astype(np.float32)
        Qbin = Q.copy()
        if Qbin.nnz:
            Qbin.data[:] = 1.0
        out_deg = np.asarray(Qbin.sum(axis=1)).ravel().astype(np.float32)
        in_deg = np.asarray(Qbin.sum(axis=0)).ravel().astype(np.float32)

        U = (Qbin + Qbin.T).tocsr()
        if U.nnz:
            U.data[:] = 1.0
            coo = U.tocoo()
            src = np.asarray(coo.row, dtype=np.int64)
            dst = np.asarray(coo.col, dtype=np.int64)
            c_sd = np.asarray(Q[src, dst]).reshape(-1).astype(np.float32)
            c_ds = np.asarray(Q[dst, src]).reshape(-1).astype(np.float32)
        else:
            src = np.empty(0, dtype=np.int64)
            dst = np.empty(0, dtype=np.int64)
            c_sd = np.empty(0, dtype=np.float32)
            c_ds = np.empty(0, dtype=np.float32)

        # Directed token lookup. U contains both orientations whenever either
        # crossing direction exists, so both (u,v) and (v,u) are available.
        token = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(src, dst))}
        cross = {}
        qcoo = Q.tocoo()
        for s, d, v in zip(qcoo.row, qcoo.col, qcoo.data):
            cross[(int(s), int(d))] = float(v)

        size_den = max(np.log1p(float(st["n_ent"])), 1.0)
        deg_den = max(np.log1p(float(max(n, 2))), 1.0)
        size_feature = (np.log1p(sizes) / size_den).astype(np.float32)

        return {
            "labels": st["labels"],
            "n_comp": n,
            "n_ent": int(st["n_ent"]),
            "size_feature": size_feature,
            "deg_den": float(deg_den),
            "out_mass": out_mass,
            "in_mass": in_mass,
            "out_deg": out_deg,
            "in_deg": in_deg,
            "src": src,
            "dst": dst,
            "c_sd": c_sd,
            "c_ds": c_ds,
            "token": token,
            "cross": cross,
        }

    def component_labels(self, qrel: int, mode: str):
        return self.templates[mode][int(qrel)]["labels"]

    @staticmethod
    def _segment_max(values: torch.Tensor, groups: torch.Tensor, n_group: int):
        out = values.new_full((n_group,), float("-inf"))
        if values.numel():
            out.scatter_reduce_(0, groups, values, reduce="amax", include_self=True)
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    def _tensor_template(self, mode: str, qrel: int, device):
        key = (mode, int(qrel), str(device))
        if key in self._device_cache:
            return self._device_cache[key]
        t = self.templates[mode][int(qrel)]
        obj = {
            "size_feature": torch.as_tensor(t["size_feature"], dtype=torch.float32, device=device),
            "out_mass": torch.as_tensor(t["out_mass"], dtype=torch.float32, device=device),
            "in_mass": torch.as_tensor(t["in_mass"], dtype=torch.float32, device=device),
            "out_deg": torch.as_tensor(t["out_deg"], dtype=torch.float32, device=device),
            "in_deg": torch.as_tensor(t["in_deg"], dtype=torch.float32, device=device),
            "src": torch.as_tensor(t["src"], dtype=torch.long, device=device),
            "dst": torch.as_tensor(t["dst"], dtype=torch.long, device=device),
            "c_sd": torch.as_tensor(t["c_sd"], dtype=torch.float32, device=device),
            "c_ds": torch.as_tensor(t["c_ds"], dtype=torch.float32, device=device),
            "labels": torch.as_tensor(t["labels"], dtype=torch.long, device=device),
        }
        self._device_cache[key] = obj
        return obj

    def pack(self, heads, rels, mode: str, device, targets=None):
        heads = np.asarray(heads, dtype=np.int64)
        rels = np.asarray(rels, dtype=np.int64)
        targets_np = None if targets is None else np.asarray(targets, dtype=np.int64)
        B = len(heads)
        if B == 0:
            raise ValueError("empty CQR batch")

        node_parts = {k: [] for k in ("size_feature", "out_mass", "in_mass", "out_deg", "in_deg")}
        src_parts, dst_parts, csd_parts, cds_parts = [], [], [], []
        node_graph_parts, edge_graph_parts, ent_map_parts = [], [], []
        head_global, gold_global = [], []
        deg_den = []
        node_offset = 0
        edge_offset = 0
        delta_node_out, delta_node_in = [], []
        delta_deg_out, delta_deg_in = [], []
        delta_csd, delta_cds = [], []

        for b, (h, r) in enumerate(zip(heads.tolist(), rels.tolist())):
            meta = self.templates[mode][int(r)]
            tt = self._tensor_template(mode, int(r), device)
            n = int(meta["n_comp"])
            e = int(len(meta["src"]))
            hc = int(meta["labels"][int(h)])

            for k in node_parts:
                node_parts[k].append(tt[k])
            src_parts.append(tt["src"] + node_offset)
            dst_parts.append(tt["dst"] + node_offset)
            csd_parts.append(tt["c_sd"])
            cds_parts.append(tt["c_ds"])
            node_graph_parts.append(torch.full((n,), b, dtype=torch.long, device=device))
            edge_graph_parts.append(torch.full((e,), b, dtype=torch.long, device=device))
            ent_map_parts.append(tt["labels"] + node_offset)
            head_global.append(node_offset + hc)
            deg_den.append(meta["deg_den"])

            if targets_np is not None:
                tc = int(meta["labels"][int(targets_np[b])])
                gold_global.append(node_offset + tc)
                if hc != tc:
                    count = float(meta["cross"].get((hc, tc), 0.0))
                    if count > 0.0:
                        delta_node_out.append(node_offset + hc)
                        delta_node_in.append(node_offset + tc)
                        if count == 1.0:
                            delta_deg_out.append(node_offset + hc)
                            delta_deg_in.append(node_offset + tc)
                        fwd = meta["token"][(hc, tc)]
                        rev = meta["token"][(tc, hc)]
                        delta_csd.append(edge_offset + fwd)
                        delta_cds.append(edge_offset + rev)

            node_offset += n
            edge_offset += e

        size_feature = torch.cat(node_parts["size_feature"], dim=0)
        out_mass = torch.cat(node_parts["out_mass"], dim=0).clone()
        in_mass = torch.cat(node_parts["in_mass"], dim=0).clone()
        out_deg = torch.cat(node_parts["out_deg"], dim=0).clone()
        in_deg = torch.cat(node_parts["in_deg"], dim=0).clone()
        src = torch.cat(src_parts, dim=0) if src_parts else torch.empty(0, dtype=torch.long, device=device)
        dst = torch.cat(dst_parts, dim=0) if dst_parts else torch.empty(0, dtype=torch.long, device=device)
        c_sd = torch.cat(csd_parts, dim=0).clone() if csd_parts else torch.empty(0, dtype=torch.float32, device=device)
        c_ds = torch.cat(cds_parts, dim=0).clone() if cds_parts else torch.empty(0, dtype=torch.float32, device=device)
        node_graph = torch.cat(node_graph_parts, dim=0)
        edge_graph = torch.cat(edge_graph_parts, dim=0) if edge_graph_parts else torch.empty(0, dtype=torch.long, device=device)

        if delta_node_out:
            idx = torch.as_tensor(delta_node_out, dtype=torch.long, device=device)
            out_mass.index_add_(0, idx, -torch.ones_like(idx, dtype=torch.float32))
        if delta_node_in:
            idx = torch.as_tensor(delta_node_in, dtype=torch.long, device=device)
            in_mass.index_add_(0, idx, -torch.ones_like(idx, dtype=torch.float32))
        if delta_deg_out:
            idx = torch.as_tensor(delta_deg_out, dtype=torch.long, device=device)
            out_deg.index_add_(0, idx, -torch.ones_like(idx, dtype=torch.float32))
        if delta_deg_in:
            idx = torch.as_tensor(delta_deg_in, dtype=torch.long, device=device)
            in_deg.index_add_(0, idx, -torch.ones_like(idx, dtype=torch.float32))
        if delta_csd:
            idx = torch.as_tensor(delta_csd, dtype=torch.long, device=device)
            c_sd.index_add_(0, idx, -torch.ones_like(idx, dtype=torch.float32))
        if delta_cds:
            idx = torch.as_tensor(delta_cds, dtype=torch.long, device=device)
            c_ds.index_add_(0, idx, -torch.ones_like(idx, dtype=torch.float32))

        out_mass.clamp_min_(0.0)
        in_mass.clamp_min_(0.0)
        out_deg.clamp_min_(0.0)
        in_deg.clamp_min_(0.0)
        c_sd.clamp_min_(0.0)
        c_ds.clamp_min_(0.0)

        deg_den_t = torch.as_tensor(deg_den, dtype=torch.float32, device=device)
        node_peak = torch.maximum(out_mass, in_mass)
        mass_peak = self._segment_max(node_peak, node_graph, B)
        mass_den = torch.log1p(torch.maximum(mass_peak, torch.ones_like(mass_peak))).clamp_min(1.0)
        x = torch.stack(
            [
                size_feature,
                torch.log1p(out_deg) / deg_den_t[node_graph],
                torch.log1p(in_deg) / deg_den_t[node_graph],
                torch.log1p(out_mass) / mass_den[node_graph],
                torch.log1p(in_mass) / mass_den[node_graph],
            ],
            dim=-1,
        )

        edge_active = ((c_sd + c_ds) > 0).to(torch.float32)
        if c_sd.numel():
            c_peak = self._segment_max(torch.maximum(c_sd, c_ds), edge_graph, B)
            cden = torch.log1p(torch.maximum(c_peak, torch.ones_like(c_peak))).clamp_min(1.0)
            l_sd = torch.log1p(c_sd) / cden[edge_graph]
            l_ds = torch.log1p(c_ds) / cden[edge_graph]
            b_sd = (c_sd > 0).to(torch.float32)
            b_ds = (c_ds > 0).to(torch.float32)
            norm_sd = c_sd / torch.sqrt(out_mass[src].clamp_min(1.0) * in_mass[dst].clamp_min(1.0))
            norm_ds = c_ds / torch.sqrt(out_mass[dst].clamp_min(1.0) * in_mass[src].clamp_min(1.0))
            rho = torch.stack([l_sd, l_ds, b_sd, b_ds, norm_sd, norm_ds], dim=-1)
        else:
            rho = torch.empty((0, 6), dtype=torch.float32, device=device)

        entity_component_index = torch.cat(ent_map_parts, dim=0).view(B, -1)
        return PackedQuotientBatch(
            x=x,
            src=src,
            dst=dst,
            rho=rho,
            edge_active=edge_active,
            node_graph=node_graph,
            edge_graph=edge_graph,
            head_index=torch.as_tensor(head_global, dtype=torch.long, device=device),
            entity_component_index=entity_component_index,
            gold_component_index=(
                torch.as_tensor(gold_global, dtype=torch.long, device=device)
                if targets_np is not None
                else None
            ),
            num_graphs=B,
            num_entities=entity_component_index.size(1),
        )
