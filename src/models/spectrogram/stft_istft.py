"""STFT/ISTFT for spectrogram models."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def stft(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
) -> torch.Tensor:
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    B, C, T = x.shape
    out = []
    for c in range(C):
        X = torch.stft(
            x[:, c],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            return_complex=True,
        )
        out.append(X)
    return torch.stack(out, dim=1)


def istft(
    X: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    length: Optional[int] = None,
) -> torch.Tensor:
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, device=X.device, dtype=torch.float32)
    B, C, F, L = X.shape
    out = []
    for c in range(C):
        x = torch.istft(
            X[:, c],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            length=length,
        )
        out.append(x)
    return torch.stack(out, dim=1)


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: Optional[int] = None,
        window: str = "hann",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        if window == "hann":
            self.register_buffer("window", torch.hann_window(self.win_length))
        else:
            self.register_buffer("window", torch.ones(self.win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
        )


class ISTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: Optional[int] = None,
        window: str = "hann",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        if window == "hann":
            self.register_buffer("window", torch.hann_window(self.win_length))
        else:
            self.register_buffer("window", torch.ones(self.win_length))

    def forward(
        self,
        X: torch.Tensor,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        return istft(
            X,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            length=length,
        )
