#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pathlib import Path

# ---------------- CONFIG ----------------
# ---------------- CONFIG ----------------
current_dir = Path(__file__).resolve().parent.parent
BASE_DIR = os.path.join(current_dir,"/examples")
OUT_DIR  = os.path.join(current_dir,"visual_and_tables")

MODELS = [
    "last","hisrep","simlpe_dct","mpt",
    "estag","hstgcnn","trajdep","pgbig","motionmixer","eqmotion","SFgDNet","sfgdnet_try"
]
PRED_SUFFIX = "_pred.pt"
TARGET_PATH = os.path.join(BASE_DIR, "last_target.pt")  # shared target for all models

HANDS = 2             # 1=first 21 joints (same as before), 2=last 21 joints
N_HAND = 21          # number of joints to show per hand
N_COLS = 10          # time samples across columns
SAMPLE_B = 0        # which sequence in batch to render
LINE_W = 2.0
COLOR_PRED = (0.05, 0.2, 0.9)    # blue
COLOR_TGT  = (0.08, 0.62, 0.08)  # green
ELEV = -20                        # 3D view elevation
AZIM = -70                       # 3D view azimuth
PADDING_FRAC = 0.05              # per-panel padding (fraction of span)
MIN_EXTENT = 1e-3                # minimal per-axis extent to avoid singular limits
# ---------------------------------------

# Pretty labels for plot
LABEL_MAP = {
    "simlpe_dct": "siMLPe",
    "SFgDNet": "SFGdNet",
    "sfgdnet_try": "SFGdNet-Try",
    "last": "Last",
    "hisrep": "HisRep",
    "mpt": "MPT",
    "estag": "Estag",
    "hstgcnn": "HSTGCNN",
    "trajdep": "TrajDep",
    "pgbig": "PGBig",
    "motionmixer": "MotionMixer",
    "eqmotion": "EqMotion"
}

# MediaPipe-like 21-joint hand skeleton:
# 0=wrist; thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20
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
    """
    Ensure shape is (B,T,N,3), then slice the requested 21-joint hand and return numpy.
    """
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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load predictions (selected 21 joints)
    loaded = {}
    n_time = None
    for m in MODELS:
        ppath = os.path.join(BASE_DIR, f"{m}{PRED_SUFFIX}")
        if not os.path.isfile(ppath):
            print(f"[WARN] missing: {ppath} — skipping")
            continue
        P = to_numpy_selected_hand(load_tensor(ppath))  # (B,T,21,3)
        if n_time is None:
            n_time = P.shape[1]  # time at axis 1
        loaded[m] = P

    if not loaded:
        raise RuntimeError("No prediction files found.")

    # Shared target (selected 21 joints)
    target = None
    if os.path.isfile(TARGET_PATH):
        target = to_numpy_selected_hand(load_tensor(TARGET_PATH))  # (B,T,21,3)
    else:
        print(f"[WARN] target not found: {TARGET_PATH}")

    # Time samples
    cols = min(N_COLS, n_time)
    ts = np.linspace(0, n_time - 1, cols).round().astype(int)  # actual frame indices shown on x-axis

    # Figure layout: model rows, separator, target row, time axis row
    n_model_rows = len(loaded)
    height_ratios = [1.0] * n_model_rows + [0.06] + [1.0] + [0.20]
    fig_w = cols * 1.2
    fig_h = sum(height_ratios) * 1.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(len(height_ratios), cols, height_ratios=height_ratios,
                  hspace=0.12, wspace=0.02)

    # Model rows
    row = 0
    for m in MODELS:
        if m not in loaded:
            continue
        label = LABEL_MAP.get(m, m)
        Pseq = loaded[m][SAMPLE_B]  # (T,N,3)
        for ci, t in enumerate(ts):
            ax = fig.add_subplot(gs[row, ci], projection="3d")
            P_frame = Pseq[t, :, :]  # (N,3) at time t
            draw_skeleton_3d(ax, P_frame, HAND_EDGES, COLOR_PRED, alpha=1.0, lw=LINE_W)
            lo, hi = bounds_with_padding(P_frame)
            format_axes_axless(ax, lo, hi)
            if ci == 0:
                ax.text2D(-0.05, 0.5, label, transform=ax.transAxes, va="center", ha="right", fontsize=10)
        row += 1

    # Separator
    for ci in range(cols):
        ax_sep = fig.add_subplot(gs[row, ci])
        ax_sep.axis("off")
        ax_sep.plot([0, 1], [0.5, 0.5], transform=ax_sep.transAxes,
                    color=(0.2, 0.2, 0.2), lw=1.0)
    row += 1

    # Target row
    if target is not None:
        Tseq = target[SAMPLE_B]  # (T,21,3)
        for ci, t in enumerate(ts):
            ax = fig.add_subplot(gs[row, ci], projection="3d")
            T_frame = Tseq[t, :, :]
            draw_skeleton_3d(ax, T_frame, HAND_EDGES, COLOR_TGT, alpha=1.0, lw=LINE_W)
            lo, hi = bounds_with_padding(T_frame)
            format_axes_axless(ax, lo, hi)
            if ci == 0:
                ax.text2D(-0.05, 0.5, "target", transform=ax.transAxes, va="center", ha="right", fontsize=10)
        row += 1

    # Time axis with frame numbers
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

    # Layout (avoid tight_layout with 3D)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.08, wspace=0.02, hspace=0.12)

    # Save
    save_path = os.path.join(OUT_DIR, "motion_comparison_hand21_3d_axless_BTN3.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()
