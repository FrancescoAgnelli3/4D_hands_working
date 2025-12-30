# hand_factorized_fixed.py
# Self-contained PyTorch implementation with auto-root detection and topological FK.

from dataclasses import dataclass
import warnings
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import torch
from torch import nn


# =========================================================
# ====================  Math / DCT  =======================
# =========================================================
def get_dct_matrix(N: int):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

@torch.jit.script
def _safe_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1, keepdim=True), min=eps))

# =========================================================
# ===================  GCN / AttModel  ====================
# =========================================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.att = nn.Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # x: (B, N, F)
        support = torch.matmul(x, self.weight)
        out = torch.matmul(self.att, support)
        if self.bias is not None:
            out = out + self.bias
        return out

class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, node_n=48, bias=True):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)
        self.drop = nn.Dropout(p_dropout)
        self.act = nn.Tanh()

    def forward(self, x):  # (B,N,F)
        y = self.gc1(x)
        B,N,F = y.shape
        y = self.bn1(y.view(B, -1)).view(B, N, F)
        y = self.act(y)
        y = self.drop(y)
        y = self.gc2(y)
        y = self.bn2(y.view(B, -1)).view(B, N, F)
        y = self.act(y)
        y = self.drop(y)
        return y + x

class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        super().__init__()
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        self.blocks = nn.ModuleList([GC_Block(hidden_feature, p_dropout, node_n=node_n) for _ in range(num_stage)])
        self.gc_out = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.drop = nn.Dropout(p_dropout)
        self.act = nn.Tanh()

    def forward(self, x, is_out_resi=True):  # (B,N,F)
        y = self.gc1(x)
        B,N,F = y.shape
        y = self.bn1(y.view(B, -1)).view(B, N, F)
        y = self.act(y)
        y = self.drop(y)
        for blk in self.blocks:
            y = blk(y)
        y = self.gc_out(y)
        if is_out_resi:
            y = y + x
        return y

class AttModel(nn.Module):
    """
    Same temporal module as your baseline:
    - local conv Q/K on (features x time)
    - DCT bottleneck + attention
    - light GCN across "nodes" where node_n = in_features (feature channels treated as nodes)
    """
    def __init__(self, in_features=48, kernel_size=10, d_model=32, num_stage=2, dct_n=10, p_dropout=0.3):
        super().__init__()
        assert kernel_size == 10, "This implementation assumes kernel_size==10."
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n

        self.convQ = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
            nn.ReLU(),
        )
        self.convK = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
            nn.ReLU(),
        )
        # IMPORTANT: node_n=in_features (each channel is treated as a "node", like the baseline)
        self.gcn = GCN(input_feature=(dct_n)*2, hidden_feature=d_model,
                       p_dropout=p_dropout, num_stage=num_stage, node_n=in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        src: (B, T_in, F) where F is flattened per-node features
        returns: (B, kernel_size+T_out, 1, F)
        """
        dct_n = self.dct_n
        src = src[:, :input_n]
        src_tmp = src.clone()
        B = src.shape[0]

        # windows for Q/K
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = get_dct_matrix(self.kernel_size + output_n)
        dev = src.device
        dct_m = torch.from_numpy(dct_m).float().to(dev)
        idct_m = torch.from_numpy(idct_m).float().to(dev)

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape([B * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(0), src_value_tmp).reshape([B, vn, dct_n, -1]).transpose(2, 3).reshape([B, vn, -1])

        idx_time = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1.0)
        for _ in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1.0)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape([B, -1, dct_n])

            input_gcn = src_tmp[:, idx_time]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            dct_out_tmp = self.gcn(dct_in_tmp)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(0), dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))

        outputs = torch.cat(outputs, dim=2)
        return outputs

# =========================================================
# ============  Kinematics / Tree construction ============
# =========================================================
def _build_undirected(num_nodes: int, links: List[Tuple[int,int]]) -> Dict[int, List[int]]:
    g = {i: [] for i in range(num_nodes)}
    for a,b in links:
        if 0 <= a < num_nodes and 0 <= b < num_nodes:
            g[a].append(b); g[b].append(a)
    return g

def _components(g: Dict[int, List[int]]) -> List[List[int]]:
    seen = set(); comps = []
    for s in g:
        if s in seen: continue
        q = [s]; seen.add(s); cur=[]
        while q:
            u = q.pop()
            cur.append(u)
            for v in g[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        comps.append(sorted(cur))
    return comps

def _choose_roots(g: Dict[int,List[int]], comps: List[List[int]], wrists_hint: Optional[List[int]] = None) -> List[int]:
    roots = []
    hint = set(wrists_hint or [])
    for comp in comps:
        # use hint if present and in this component
        cand = [h for h in hint if h in comp]
        if cand:
            roots.append(cand[0]); continue
        # fallback: node with highest degree in comp (hub)
        root = max(comp, key=lambda x: len(g[x]))
        roots.append(root)
    return roots

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

# =========================================================
# ================  Kinematics utilities =================
# =========================================================
def compute_dirs_lens(x: torch.Tensor, parents: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B,T,N,3)
    returns dirs (B,T,N,3), lens (B,T,N,1)
    """
    B,T,N,_ = x.shape
    dirs = torch.zeros_like(x)
    lens = torch.zeros(B,T,N,1, device=x.device, dtype=x.dtype)
    for c,p in enumerate(parents):
        if p == -1: continue
        v = x[:,:,c] - x[:,:,p]
        l = _safe_norm(v)
        u = v / l
        dirs[:,:,c] = u
        lens[:,:,c] = l
    return dirs, lens

def fk_topological(
    dirs: torch.Tensor,        # (B,T,N,3) unit directions
    lens: torch.Tensor,        # (B,T,N,1) or (B,T,N)
    parents: List[int],        # length N, parent index or -1 for root
    order: List[int],          # topological order over 0..N-1
    roots: List[int],          # indices of root joints (e.g., wrists)
) -> torch.Tensor:
    """
    Forward kinematics with all roots fixed at (0,0,0), regardless of wrist_pos.

    Args:
        wrist_pos: unused (kept to match the original signature).
        dirs: (B,T,N,3) unit vectors pointing child -> away from parent.
        lens: (B,T,N,1) or (B,T,N) bone lengths.
        parents: list of ints; -1 for roots, otherwise parent index.
        order: a valid topological order of nodes [0..N-1].
        roots: list of node indices that act as roots (set to (0,0,0)).

    Returns:
        xyz: (B,T,N,3) absolute joint coordinates with roots at the origin.
    """
    assert dirs.dim() == 4 and dirs.size(-1) == 3, "dirs must be (B,T,N,3)"
    B, T, N, _ = dirs.shape
    assert len(parents) == N, "parents length must equal N"
    assert all(0 <= u < N for u in order) and len(order) == N, "invalid topo order"
    assert all((p == -1) or (0 <= p < N) for p in parents), "invalid parents"

    if lens.dim() == 3:
        lens = lens.unsqueeze(-1)  # -> (B,T,N,1)
    assert lens.shape == (B, T, N, 1), "lens must be (B,T,N,1) or (B,T,N)"

    # Allocate output
    xyz = torch.zeros_like(dirs)  # (B,T,N,3), device/dtype matched

    # Pin all roots at the origin (0,0,0)
    if len(roots) > 0:
        roots_tensor = torch.as_tensor(roots, device=dirs.device)
        xyz.index_fill_(2, roots_tensor, 0.0)

    # One pass in topological order
    # Skip true roots (parents[u] == -1) — they’re already zeroed.
    for u in order:
        p = parents[u]
        if p == -1:
            continue
        # xyz[..., u, :] = xyz[..., p, :] + dirs[..., u, :] * lens[..., u, :]
        xyz[:, :, u] = xyz[:, :, p] + dirs[:, :, u] * lens[:, :, u]

    return xyz

# =========================================================
# ===================  Models / Config  ===================
# =========================================================
@dataclass
class HisRepConfig:
    num_nodes: int
    input_length: int
    pred_length: int
    hidden_dim: int = 256
    num_stages: int = 2
    dct_n: int = 10
    kernel_size: int = 10
    p_dropout: float = 0.3
    enable_delta_length: bool = True
    delta_length_scale: float = 0.001
    # graph / wrists
    wrists_hint: Optional[List[int]] = None  # e.g., [5,26]; if None, auto-detect

class HisRepForecaster(nn.Module):
    """
    - Detects roots (wrists) or uses wrists_hint.
    - Head A: predicts wrists (pure temporal).
    - Head B: predicts unit directions for all distal joints.
    - Optional tiny Δ-lengths.
    - FK in topological order.
    """
    def __init__(self, cfg: HisRepConfig):
        super().__init__()
        max_coeffs = int(cfg.kernel_size + cfg.pred_length)
        raw_dct = cfg.dct_n if cfg.dct_n is not None else max_coeffs
        clamped_dct = max(1, min(int(raw_dct), max_coeffs))
        if clamped_dct != raw_dct:
            warnings.warn(
                f"HisRepHand: clamping dct_n from {raw_dct} to {clamped_dct} to respect kernel_size + pred_length "
                f"({cfg.kernel_size}+{cfg.pred_length})",
                RuntimeWarning,
            )
        cfg.dct_n = clamped_dct

        self.cfg = cfg
        links = bone_link_1 + bone_link_2
        self.parents, self.order, self.roots, self.edges = build_tree(cfg.num_nodes, links, cfg.wrists_hint)

        missing = [i for i,p in enumerate(self.parents) if p==-1 and i not in self.roots]
        assert len(missing)==0, f"Unreached nodes (check wrists and links): {missing}"

        # indices
        self.root_set = set(self.roots)
        self.distal_idx = [i for i in range(cfg.num_nodes) if i not in self.root_set]
        self.num_distal = len(self.distal_idx)
        self.R = len(self.roots)

        # Head B: articulation (unit directions)
        self.artic_backbone = AttModel(
            in_features=self.num_distal * 3,
            kernel_size=cfg.kernel_size,
            d_model=cfg.hidden_dim,
            num_stage=cfg.num_stages,
            dct_n=cfg.dct_n,
            p_dropout=cfg.p_dropout
        )

        # Optional Δ-length head
        self.len_head = nn.Linear(self.num_distal * 3, self.num_distal) if cfg.enable_delta_length else None

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor:
        """
        x: (B,T_in,N,3)
        returns: (B,T_out,N,3)
        """
        B, T_in, N, C = x.shape
        assert N == self.cfg.num_nodes

        required_history = self.cfg.pred_length + self.artic_backbone.kernel_size
        if T_in < required_history:
            pad_len = required_history - T_in
            pad = x[:, -1:].expand(B, pad_len, N, C)
            x = torch.cat([x, pad], dim=1)
            T_in = required_history

        with torch.no_grad():
            dirs_in, lens_in = compute_dirs_lens(x, self.parents)  # (B,T_in,N,3), (B,T_in,N,1)

        # ==== Head B: directions ====
        d_in = dirs_in[:, :, self.distal_idx]                                 # (B,T_in,D,3)
        d_in = d_in.reshape(B, T_in, self.num_distal * 3)
        d_out = self.artic_backbone(d_in, output_n=self.cfg.pred_length, input_n=T_in, itera=1)
        d_out_flat = d_out[:, self.cfg.kernel_size:, 0]                       # (B,T_out,D*3)
        dirs_future = d_out_flat.reshape(B, self.cfg.pred_length, self.num_distal, 3)
        dirs_future = dirs_future / (dirs_future.norm(dim=-1, keepdim=True) + 1e-8)

        # ==== Lengths ====
        last_len = lens_in[:, -1]                                             # (B,N,1)
        last_len_dist = last_len[:, self.distal_idx, 0]                       # (B,D)
        if self.len_head is not None:
            dl = self.len_head(d_out_flat)                                    # (B,T_out,D)
            lens_future_dist = (last_len_dist.unsqueeze(1) +
                                self.cfg.delta_length_scale * dl).clamp_min(1e-6)
        else:
            lens_future_dist = last_len_dist.unsqueeze(1).expand(B, self.cfg.pred_length, self.num_distal)

        # pack full tensors
        dirs_full = torch.zeros(B, self.cfg.pred_length, N, 3, device=x.device, dtype=x.dtype)
        lens_full = torch.zeros(B, self.cfg.pred_length, N, 1, device=x.device, dtype=x.dtype)
        dirs_full[:, :, self.distal_idx] = dirs_future
        lens_full[:, :, self.distal_idx, 0] = lens_future_dist

        # ==== FK (topological, once) ====
        xyz_future = fk_topological(dirs_full, lens_full,
                                    self.parents, self.order, self.roots)     # (B,T_out,N,3)

        if return_aux:
            # pack full-N predicted directions (zeros at roots)
            pred_dirs_full = torch.zeros_like(dirs_full)
            pred_dirs_full[:, :, self.distal_idx] = dirs_future

            # predicted wrists (B, T_out, R, 3)
            pred_wrists = xyz_future[:, :, self.roots, :]

            # predicted Δlengths if enabled
            delta_lengths = dl if (self.len_head is not None) else None

            aux = {
                "pred_dirs_full": pred_dirs_full,   # (B,T_out,N,3)
                "pred_wrists": pred_wrists,         # (B,T_out,R,3)
                "delta_lengths": delta_lengths,     # (B,T_out,D) or None
            }
            return xyz_future, aux
        return xyz_future

# =========================================================
# ===================  Loss (optional) ====================
# =========================================================
class FactorizedLoss(nn.Module):
    def __init__(self,
                 lambda_pos=0.5,
                 lambda_dir=2.0,
                 lambda_unit=0.1,
                 lambda_len=0.01,
                 lambda_jerk=0.001):
        super().__init__()
        self.lp = lambda_pos
        self.ld = lambda_dir
        self.lu = lambda_unit
        self.ll = lambda_len
        self.lj = lambda_jerk

    @staticmethod
    def _finite_diff3(x: torch.Tensor) -> torch.Tensor:
        return x[:, 3:] - 3 * x[:, 2:-1] + 3 * x[:, 1:-2] - x[:, :-3]

    def forward(self,
                pred_xyz: torch.Tensor,      # (B,T_out,N,3)
                gt_xyz: torch.Tensor,        # (B,T_out,N,3)
                pred_dirs: torch.Tensor,     # (B,T_out,N,3) (put zeros for roots)
                gt_dirs: torch.Tensor,       # (B,T_out,N,3)
                dir_mask: torch.Tensor,      # (N,)
                pred_wrists: torch.Tensor,   # (B,T_out,R,3)
                delta_lengths: Optional[torch.Tensor] = None  # (B,T_out,D)
                ):
        l_pos = torch.nn.functional.l1_loss(pred_xyz, gt_xyz)

        mask = dir_mask.view(1,1,-1,1)
        pd = pred_dirs * mask
        gd = gt_dirs * mask
        cos = torch.nn.functional.cosine_similarity(pd, gd, dim=-1)
        l_dir = (1.0 - cos).mean()

        # l_unit = (pred_dirs.norm(dim=-1) - 1.0).abs().mean()
        mask = dir_mask.view(1,1,-1,1)                     # already computed
        l_unit = ((pred_dirs.norm(dim=-1, keepdim=True) - 1.0).abs() * mask).mean()


        l_len = (delta_lengths**2).mean() if delta_lengths is not None else 0.0

        jerk = self._finite_diff3(pred_wrists)
        l_jerk = (jerk**2).mean()

        return self.lp*l_pos + self.ld*l_dir + self.lu*l_unit + self.ll*l_len + self.lj*l_jerk
