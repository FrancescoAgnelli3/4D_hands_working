import os
import argparse
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


def _load_array(path: str, sample: int | None = 0) -> np.ndarray:
    """Load .npy or .pt and return (T, N, 3) float32 numpy array.

    - Accepts batched inputs (B, T, N, 3) and selects the `sample` index.
    - If `sample` is None and B==1, auto-squeezes to (T, N, 3).
    """
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        t = torch.load(path, map_location="cpu")
        # Common dump formats: raw tensor, or dict with a key
        if isinstance(t, dict):
            for k in ("data", "pred", "target", "gt", "prediction"):
                if k in t:
                    t = t[k]
                    break
        arr = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    arr = np.asarray(arr)
    # Handle batched inputs
    if arr.ndim == 4 and arr.shape[-1] == 3:
        B = arr.shape[0]
        if sample is None:
            if B == 1:
                arr = arr[0]
            else:
                raise ValueError(f"Array is batched with B={B}; please set --sample index.")
        else:
            if not (0 <= int(sample) < B):
                raise IndexError(f"--sample {sample} out of range for batch size {B}")
            arr = arr[int(sample)]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (T, N, 3), got {arr.shape} from {path}")
    return arr.astype(np.float32)


class PlotHands:
    """3D animation comparing prediction vs GT for 42-node two-hand graph.

    Expects arrays shaped (T, N, 3). N should be 42 for two hands.
    """

    def __init__(self, prediction_data: np.ndarray, gt_data: np.ndarray, interval_ms: int = 40):
        assert prediction_data.shape == gt_data.shape, "Prediction and GT must have same shape"
        assert prediction_data.ndim == 3 and prediction_data.shape[-1] == 3
        self.pred = prediction_data
        self.gt = gt_data
        self.nframes = prediction_data.shape[0]
        self.interval = int(interval_ms)

        # Hand graph edges (no self-loops). Matches model_4GRU.Graph
        bone_link_1 = [
            (4, 19), (3, 16), (2, 13), (1, 10), (19, 18), (16, 15), (13, 12), (10, 9),
            (18, 17), (15, 14), (12, 11), (9, 8), (17, 5), (14, 5), (11, 5), (8, 5),
            (0, 7), (7, 6), (6, 5), (20, 5), (17, 14), (14, 11), (11, 8)
        ]
        bone_link_2 = [
            (25, 40), (24, 37), (23, 34), (22, 31), (40, 39), (37, 36), (34, 33), (31, 30),
            (39, 38), (36, 35), (33, 32), (30, 29), (38, 26), (35, 26), (32, 26), (29, 26),
            (21, 28), (28, 27), (27, 26), (41, 26), (38, 35), (35, 32), (32, 29)
        ]
        self.edges = bone_link_1 + bone_link_2

        # Axis limits from data with margin
        both = np.concatenate([self.pred.reshape(-1, 3), self.gt.reshape(-1, 3)], axis=0)
        mins = both.min(axis=0)
        maxs = both.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        center = 0.5 * (mins + maxs)
        half = 0.6 * span.max()  # square box

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self._lines = []

    def _clear_lines(self):
        for ln in self._lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._lines = []

    def _plot_edges(self, xyz: np.ndarray, color: str, lw: float = 2.0):
        for i, j in self.edges:
            xs = [xyz[i, 0], xyz[j, 0]]
            ys = [xyz[i, 1], xyz[j, 1]]
            zs = [xyz[i, 2], xyz[j, 2]]
            self._lines.append(self.ax.plot(xs, ys, zs, color=color, linewidth=lw)[0])

    def update(self, frame: int):
        self._clear_lines()
        gt = self.gt[frame]
        pr = self.pred[frame]
        # Prediction in red, GT in blue
        self._plot_edges(pr, color="#f94e3e", lw=2.0)
        self._plot_edges(gt, color="#0780ea", lw=2.0)
        return self._lines

    def animate(self):
        # Keep a strong reference on self
        self.ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=self.nframes, interval=self.interval,
            blit=False, repeat=False
        )
        return self.ani

def main():
    parser = argparse.ArgumentParser(description="Visualize two-hand graph prediction vs GT")
    parser.add_argument(
    "--fps", type=float, default=2,
    help="Frames per second for saved animation (lower = slower). If omitted, derived from --interval.")
    parser.add_argument("--pred", type=str, default=os.path.join(os.path.dirname(__file__), "examples", "hisrep_hand_space_True_pred.pt"))
    parser.add_argument("--gt", type=str, default=os.path.join(os.path.dirname(__file__), "examples", "hisrep_hand_space_True_target.pt"))
    parser.add_argument("--out", type=str, default=None, help="Output GIF/MP4 path; auto-saves if backend is non-interactive")
    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--sample", type=int, default=4)
    args = parser.parse_args()

    pred = _load_array(args.pred, sample=args.sample)
    gt = _load_array(args.gt, sample=args.sample)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch pred {pred.shape} vs gt {gt.shape}")

    viz = PlotHands(prediction_data=pred, gt_data=gt, interval_ms=args.interval)
    ani = viz.animate()


    interactive = plt.isinteractive()  # or your _interactive_backend()

    out_path = args.out
    if out_path is None and not interactive:
        out_path = os.path.join(os.getcwd(), "hands_animation.gif")

    if out_path:
        # <<< NEW: choose a low FPS for a slower GIF >>>
        base_fps = max(1.0, 1000.0 / args.interval)  # derive from interval if user doesn't specify
        fps = getattr(args, "fps", None) or base_fps  # allow CLI override
        writer = PillowWriter(fps=fps)
        ani.save(out_path, writer=writer)  # slower if fps is small (e.g., 2–5)
        print(f"Saved: {out_path} @ {fps:.2f} fps")
        plt.close(viz.fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()