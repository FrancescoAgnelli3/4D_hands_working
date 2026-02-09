#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ---------------- CONFIG ----------------
current_dir = Path(__file__).resolve().parent.parent
BASE_DIR = os.path.join(current_dir, "examples")
OUT_DIR = os.path.join(current_dir, "visual_and_tables")

MODELS = [
    "last","hisrep","simlpe_dct","mpt","twostage_dct_diffusion","trajdep","pgbig","eqmotion"
]
PRED_SUFFIX = "_pred.pt"
TARGET_PATH = os.path.join(BASE_DIR, "twostage_dct_diffusion_target.pt")  # shared target for all models

HANDS = 1
N_HAND = 21
N_COLS = 10
SAMPLE_B = 1
LINE_W = 2.0
COLOR_PRED = (0.05, 0.2, 0.9)
COLOR_TGT  = (0.08, 0.62, 0.08)
ELEV = -50
AZIM = -50
PADDING_FRAC = 0.05
MIN_EXTENT = 1e-3
EPS_SCALE = 1e-12
# ---------------------------------------

LABEL_MAP = {
    "spectral": "Spectral",
    "simlpe_dct": "siMLPe",
    "SFgDNet": "SFGdNet",
    "SplineEqNet": "SplineEqNet",
    "last": "Last",
    "hisrep": "HisRep",
    "mpt": "MPT",
    "estag": "Estag",
    "hstgcnn": "HSTGCNN",
    "trajdep": "TrajDep",
    "pgbig": "PGBig",
    "motionmixer": "MotionMixer",
    "eqmotion": "EqMotion",
    "physmamba": "PhysMamba",
    "simlpe_wave": "SimlpeWave",
    "twostage_dct_diffusion": "2S-DCT-Diffusion",
}

HAND_EDGES = [(4,19),(3,16),(2,13),(1,10),(19,18),(16,15),(13,12),(10,9),
              (18,17),(15,14),(12,11),(9,8),(17,5),(14,5),(11,5),(8,5),
              (0,7),(7,6),(6,5),(20,5),(17,14),(14,11),(11,8)]

def load_tensor(path):
    x = torch.load(path, map_location="cpu")
    for k in ["pred", "prediction", "preds", "output"]:
        if isinstance(x, dict) and k in x:
            x = x[k]
            break
    return x  # expected B,T,N,3

def to_numpy_selected_hand(x, hand=HANDS):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() != 4 or x.shape[-1] != 3:
        raise ValueError(f"Tensor must be B×T×N×3; got {tuple(x.shape)}")
    if hand == 1:
        sl = slice(0, N_HAND)
    elif hand == 2:
        if x.shape[2] < N_HAND:
            raise ValueError(f"Tensor has only {x.shape[2]} joints")
        sl = slice(-N_HAND, None)
    else:
        raise ValueError("hand must be 1 or 2")
    x = x[:, :, sl, :]
    return x.detach().cpu().numpy()  # (B,T,21,3)

def draw_skeleton_3d(ax, P, edges, color, alpha=1.0, lw=2.0):
    for i, j in edges:
        if i < P.shape[0] and j < P.shape[0]:
            xi, yi, zi = P[i]
            xj, yj, zj = P[j]
            ax.plot([xi, xj], [yi, yj], [zi, zj], lw=lw, color=color, alpha=alpha)

def bounds_with_padding(P, pad_frac=PADDING_FRAC, min_extent=MIN_EXTENT):
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    span = hi - lo
    span = np.maximum(span, min_extent)
    pad = span * pad_frac
    lo2 = lo - pad
    hi2 = lo + span + pad
    return lo2, hi2

def format_axes_axless(ax, lo, hi):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=ELEV, azim=AZIM)
    extent = hi - lo
    extent = np.maximum(extent, 1e-6)
    ax.set_box_aspect(extent)
    ax.set_axis_off()

def frame_scale_only(P):
    """
    A scale proxy that does NOT translate: RMS distance of joints from origin.
    """
    return float(np.sqrt(np.mean(np.sum(P * P, axis=1))) + EPS_SCALE)

def scale_pred_to_target_only(P_pred, P_tgt):
    """
    Scale P_pred so that its RMS-from-origin matches P_tgt at the same time.
    No translation is applied.
    """
    sp = frame_scale_only(P_pred)
    st = frame_scale_only(P_tgt)
    s = st / sp
    return P_pred * s

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Target is required for "corresponding in time target" scaling
    if not os.path.isfile(TARGET_PATH):
        raise RuntimeError(f"Target not found: {TARGET_PATH}")
    target = to_numpy_selected_hand(load_tensor(TARGET_PATH))  # (B,T,21,3)

    # Load predictions
    loaded = {}
    n_time = target.shape[1]
    for m in MODELS:
        ppath = os.path.join(BASE_DIR, f"{m}{PRED_SUFFIX}")
        if not os.path.isfile(ppath):
            print(f"[WARN] missing: {ppath} — skipping")
            continue
        P = to_numpy_selected_hand(load_tensor(ppath))  # (B,T,21,3)
        if P.shape[1] != n_time:
            raise ValueError(f"Time mismatch for {m}: pred T={P.shape[1]} vs target T={n_time}")
        loaded[m] = P

    if not loaded:
        raise RuntimeError("No prediction files found.")

    # Time samples (columns)
    cols = min(N_COLS, n_time)
    ts = np.linspace(0, n_time - 1, cols).round().astype(int)

    # Layout
    n_model_rows = sum(1 for m in MODELS if m in loaded)
    height_ratios = [1.0] * n_model_rows + [0.06] + [1.0] + [0.20]
    fig_w = cols * 1.2
    fig_h = sum(height_ratios) * 1.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(len(height_ratios), cols, height_ratios=height_ratios,
                  hspace=0.12, wspace=0.02)

    # Precompute per-column target bounds so every subplot in that column shares the same view
    tgt_seq = target[SAMPLE_B]  # (T,21,3)
    col_bounds = {}
    for t in ts:
        T_frame = tgt_seq[t]
        lo, hi = bounds_with_padding(T_frame)
        col_bounds[int(t)] = (lo, hi)

    # Model rows: scale each prediction frame to the corresponding target frame (same t), scale-only
    row = 0
    for m in MODELS:
        if m not in loaded:
            continue
        label = LABEL_MAP.get(m, m)
        Pseq = loaded[m][SAMPLE_B]  # (T,21,3)
        for ci, t in enumerate(ts):
            ax = fig.add_subplot(gs[row, ci], projection="3d")
            T_frame = tgt_seq[t]
            P_frame = Pseq[t]

            P_scaled = scale_pred_to_target_only(P_frame, T_frame)
            draw_skeleton_3d(ax, P_scaled, HAND_EDGES, COLOR_PRED, alpha=1.0, lw=LINE_W)

            lo, hi = col_bounds[int(t)]
            format_axes_axless(ax, lo, hi)

            if ci == 0:
                ax.text2D(-0.05, 0.5, label, transform=ax.transAxes,
                          va="center", ha="right", fontsize=10)
        row += 1

    # Separator
    for ci in range(cols):
        ax_sep = fig.add_subplot(gs[row, ci])
        ax_sep.axis("off")
        ax_sep.plot([0, 1], [0.5, 0.5], transform=ax_sep.transAxes,
                    color=(0.2, 0.2, 0.2), lw=1.0)
    row += 1

    # Target row (unscaled)
    for ci, t in enumerate(ts):
        ax = fig.add_subplot(gs[row, ci], projection="3d")
        T_frame = tgt_seq[t]
        draw_skeleton_3d(ax, T_frame, HAND_EDGES, COLOR_TGT, alpha=1.0, lw=LINE_W)

        lo, hi = col_bounds[int(t)]
        format_axes_axless(ax, lo, hi)

        if ci == 0:
            ax.text2D(-0.05, 0.5, "target", transform=ax.transAxes,
                      va="center", ha="right", fontsize=10)
    row += 1

    # Time axis
    ax_time = fig.add_subplot(gs[row, :])
    for k in ["top", "right", "left"]:
        ax_time.spines[k].set_visible(False)
    ax_time.set_yticks([])
    ax_time.set_xlim(-0.5, cols - 0.5)
    ax_time.set_xticks(np.arange(cols))
    ax_time.set_xticklabels([str(int(f)) for f in ts])
    ax_time.set_xlabel("frame", labelpad=6)
    ax_time.annotate("", xy=(cols - 1 + 0.12, 0), xytext=(-0.12, 0),
                     arrowprops=dict(arrowstyle="->", lw=1.2),
                     xycoords=("data", "axes fraction"))

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.08,
                        wspace=0.02, hspace=0.12)

    save_path = os.path.join(OUT_DIR, "motion_comparison_hand21_3d_axless_BTN3_scaled_to_target_only.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")

if __name__ == "__main__":
    main()
