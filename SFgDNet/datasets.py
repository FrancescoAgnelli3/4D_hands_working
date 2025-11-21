import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union
import numpy as np

class AssemblyDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(self, sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]], input_n: int, output_n: int, stride: int = 1):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))

        print(
            "AssemblyDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        return seq[a:b], seq[b:c], norm_factor

class h2oDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(self, sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]], input_n: int, output_n: int, stride: int = 1):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))

        print(
            "h2oDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        return seq[a:b], seq[b:c], norm_factor

class BigHandsDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(self, sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]], input_n: int, output_n: int, stride: int = 1):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))

        print(
            "BigHandsDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        return seq[a:b], seq[b:c], norm_factor

class FPHADataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(self, sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]], input_n: int, output_n: int, stride: int = 1):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))

        print(
            "FPHADataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        return seq[a:b], seq[b:c], norm_factor


