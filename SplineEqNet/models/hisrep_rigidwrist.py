# =========================================================
# ==========  Wrist-only forecast with rigid hand =========
# =========================================================
# Depends on your existing: AttModel, build_tree, _build_undirected, bone_link_1/2
# Input/Output API matches your forecaster: x: (B,T_in,N,3) -> (B,T_out,N,3)

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from torch import nn
from .hisrep_hand import AttModel
from typing import List, Tuple, Optional, Dict


bone_link_1 = [(4,19),(3,16),(2,13),(1,10),(19,18),(16,15),(13,12),(10,9),
               (18,17),(15,14),(12,11),(9,8),(17,5),(14,5),(11,5),(8,5),
               (0,7),(7,6),(6,5),(20,5),(17,14),(14,11),(11,8)]
bone_link_2 = [(25,40),(24,37),(23,34),(22,31),(40,39),(37,36),(34,33),(31,30),
               (39,38),(36,35),(33,32),(30,29),(38,26),(35,26),(32,26),(29,26),
               (21,28),(28,27),(27,26),(41,26),(38,35),(35,32),(32,29)]


def _build_undirected(num_nodes: int, links: List[Tuple[int,int]]) -> Dict[int, List[int]]:
    g = {i: [] for i in range(num_nodes)}
    for a,b in links:
        if 0 <= a < num_nodes and 0 <= b < num_nodes:
            g[a].append(b); g[b].append(a)
    return g


def build_tree(num_nodes: int,
               links: List[Tuple[int,int]],
               wrists_hint: Optional[List[int]] = None):
    """
    Build a directed kinematic tree by BFS from one root per connected component.
    Returns:
      parents: list length N with -1 for roots
      order: topological order for FK (parents before children)
      roots: list of roots (wrists)
      edges: list of directed (parent, child)
    """
    g = _build_undirected(num_nodes, links)
    comps = _components(g)
    roots = _choose_roots(g, comps, wrists_hint)
    parents = [-1]*num_nodes
    order: List[int] = []
    edges: List[Tuple[int,int]] = []

    for r in roots:
        q = [r]; parents[r] = -1
        while q:
            u = q.pop(0)
            order.append(u)
            for v in g[u]:
                if v == parents[u]:  # skip back edge
                    continue
                if parents[v] == -1 and v not in roots:  # not assigned and not another root
                    parents[v] = u
                    edges.append((u,v))
                    q.append(v)
    # Keep only nodes that were reached from roots (others keep -1 but won’t be used)
    return parents, order, roots, edges


@dataclass
class WristOnlyConfig:
    num_nodes: int
    input_length: int
    pred_length: int
    hidden_dim: int = 256
    num_stages: int = 2
    dct_n: int = 10
    kernel_size: int = 10
    p_dropout: float = 0.3
    wrists_hint: Optional[List[int]] = None  # e.g., [5, 26]

class WristRigidForecaster(nn.Module):
    """
    Predict future wrist positions using the same AttModel stack;
    for every other joint, reuse its last-frame offset relative to its wrist (per-hand root).
    """
    def __init__(self, cfg: WristOnlyConfig):
        super().__init__()
        self.cfg = cfg
        links = bone_link_1 + bone_link_2

        # Build hand components and choose roots (wrists)
        parents, order, roots, edges = build_tree(cfg.num_nodes, links, cfg.wrists_hint)
        self.roots = roots                    # list of node ids that are wrists
        self.R = len(self.roots)
        assert self.R >= 1, "At least one wrist/root must be detected."
        self.register_buffer("root_idx", torch.tensor(self.roots, dtype=torch.long))

        # Build a node->root index map (which wrist a node belongs to)
        g = _build_undirected(cfg.num_nodes, links)
        self.comp_nodes: List[List[int]] = []
        node2root = torch.full((cfg.num_nodes,), -1, dtype=torch.long)
        for ridx, r in enumerate(self.roots):
            # BFS to collect component nodes reachable from this root
            seen = {r}
            q = [r]
            nodes = [r]
            while q:
                u = q.pop(0)
                for v in g[u]:
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
                        nodes.append(v)
            nodes = sorted(nodes)
            self.comp_nodes.append(nodes)
            for n in nodes:
                node2root[n] = ridx

        assert (node2root >= 0).all(), "Some nodes are not assigned to any wrist/component."

        # Register constant buffers for fast gather at runtime
        self.register_buffer("node2root", node2root)  # (N,)

        # Wrist temporal backbone (same architecture as your model)
        self.wrist_backbone = AttModel(
            in_features=self.R * 3,
            kernel_size=cfg.kernel_size,
            d_model=max(128, cfg.hidden_dim // 2),
            num_stage=max(1, cfg.num_stages - 1),
            dct_n=cfg.dct_n,
            p_dropout=cfg.p_dropout,
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        x: (B, T_in, N, 3) absolute coordinates
        returns:
            xyz_future: (B, T_out, N, 3)
            aux (optional): dict with 'pred_wrists' (B, T_out, R, 3) and 'last_offsets' (B, N, 3)
        """
        B, T_in, N, _ = x.shape
        assert N == self.cfg.num_nodes

        # ---- 1) Predict wrists with the same temporal module ----
        wrist_idx = self.root_idx
        wrists_in = torch.index_select(x, dim=2, index=wrist_idx)  # (B, T_in, R, 3)
        wr_flat = wrists_in.reshape(B, T_in, self.R * 3)
        wr_out = self.wrist_backbone(
            wr_flat, output_n=self.cfg.pred_length, input_n=T_in, itera=1
        )  # (B, kernel+T_out, 1, R*3)
        wr_out_flat = wr_out[:, self.cfg.kernel_size:, 0]                  # (B, T_out, R*3)
        wrists_future = wr_out_flat.reshape(B, self.cfg.pred_length, self.R, 3)  # (B, T_out, R, 3)

        # ---- 2) Compute last-frame offsets (per node) relative to its wrist ----
        # last observed wrists: (B, 1, R, 3)
        wrists_last = wrists_in[:, -1:, :, :]  # (B, 1, R, 3)

        # Map per-node to its wrist coordinate at the last frame
        # Build gather index: (1,1,N,1) -> expand to (B,1,N,3)
        idx = self.node2root.view(1, 1, N, 1).to(x.device)
        wrists_last_per_node = torch.gather(
            wrists_last, dim=2, index=idx.expand(B, 1, N, 3)
        )  # (B, 1, N, 3)

        # Node absolute last positions: (B,1,N,3)
        x_last = x[:, -1].unsqueeze(1)

        # Fixed offsets to carry forward
        last_offsets = (x_last - wrists_last_per_node).squeeze(1)          # (B, N, 3)
        # For wrist nodes, the offset is (near) zero already; no special-case needed.

        # ---- 3) Broadcast predicted wrists to nodes per their component, then add offsets ----
        # wrists_future: (B, T_out, R, 3) -> gather per-node root
        idx_full = self.node2root.view(1, 1, N, 1).to(x.device)
        wrists_future_per_node = torch.gather(
            wrists_future, dim=2, index=idx_full.expand(B, self.cfg.pred_length, N, 3)
        )  # (B, T_out, N, 3)

        # Add fixed last-frame offsets
        xyz_future = wrists_future_per_node + last_offsets.unsqueeze(1)     # (B, T_out, N, 3)

        if return_aux:
            aux = {
                "pred_wrists": wrists_future,  # (B, T_out, R, 3)
                "last_offsets": last_offsets,  # (B, N, 3)
            }
            return xyz_future, aux
        return xyz_future
