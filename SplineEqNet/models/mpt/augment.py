import torch
import torch.nn as nn

class TimeCutout1D(nn.Module):
    """
    Zero a random contiguous chunk along time for each sample (causal-safe).
    Applies with probability p; chunk length is a fraction in [frac_min, frac_max] of T.
    """
    def __init__(self, p: float = 0.5, frac_min: float = 0.05, frac_max: float = 0.2):
        super().__init__()
        assert 0 <= p <= 1 and 0 <= frac_min <= frac_max <= 0.99
        self.p = float(p)
        self.frac_min = float(frac_min)
        self.frac_max = float(frac_max)

    @torch.no_grad()
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (B, T, N, C)
        if not self.training or self.p == 0:
            return Z
        B, T, N, C = Z.shape
        device = Z.device
        out = Z
        # Decide which samples to augment
        do = torch.rand(B, device=device) < self.p
        if not do.any():
            return out
        # For each selected sample, pick a random chunk length & start
        frac = torch.empty(B, device=device).uniform_(self.frac_min, self.frac_max)
        L = (frac * T).clamp(min=1).to(torch.long)
        # ensure the cutout lies fully in the observed past (<= last index)
        start_max = (T - L).clamp(min=0)
        start = torch.zeros(B, dtype=torch.long, device=device)
        nonzero = start_max > 0
        start[nonzero] = torch.randint(low=0, high=int(start_max.max().item()) + 1, size=(nonzero.sum().item(),), device=device)
        # Mask out per-sample contiguous ranges
        for b in torch.where(do)[0].tolist():
            s = int(min(start[b].item(), T - L[b].item()))
            e = s + int(L[b].item())
            out[b, s:e, :, :] = 0
        return out


class NodeDropout(nn.Module):
    """
    Drop entire nodes (all channels) across the whole window with prob p_node per node.
    """
    def __init__(self, p_node: float = 0.2):
        super().__init__()
        assert 0 <= p_node < 1
        self.p_node = float(p_node)

    @torch.no_grad()
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_node == 0:
            return Z
        B, T, N, C = Z.shape
        device = Z.device
        keep = torch.bernoulli(torch.full((B, N), 1.0 - self.p_node, device=device))  # (B,N)
        keep = keep.view(B, 1, N, 1)  # broadcast over time & channels
        return Z * keep


class ChannelDropout(nn.Module):
    """
    Drop feature channels across the whole window with prob p_ch per channel.
    """
    def __init__(self, p_ch: float = 0.1):
        super().__init__()
        assert 0 <= p_ch < 1
        self.p_ch = float(p_ch)

    @torch.no_grad()
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_ch == 0:
            return Z
        B, T, N, C = Z.shape
        device = Z.device
        keep = torch.bernoulli(torch.full((B, C), 1.0 - self.p_ch, device=device))  # (B,C)
        keep = keep.view(B, 1, 1, C)  # broadcast over time & nodes
        return Z * keep


class AugmentStack(nn.Module):
    """
    Compose the three in a single call. Order matters: node/channel masks first, then time cutout.
    """
    def __init__(self, p_node=0.2, p_ch=0.1, p_time=0.5, frac_min=0.05, frac_max=0.2):
        super().__init__()
        self.node = NodeDropout(p_node)
        self.chan = ChannelDropout(p_ch)
        self.time = TimeCutout1D(p=p_time, frac_min=frac_min, frac_max=frac_max)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # no-grad inside modules; detach/reattach so we don’t block autograd
        Z = self.node(Z)
        Z = self.chan(Z)
        Z = self.time(Z)
        return Z
