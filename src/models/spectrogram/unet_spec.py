"""U-Net spectrogram vocal separation (magnitude mask, mixture phase)."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stft_istft import STFT, ISTFT


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, activation: str = "leaky_relu"):
        super().__init__()
        act = nn.LeakyReLU(0.2, inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2DSpec(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.depth = depth
        ch = [min(base_channels * (2**i), 256) for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for i in range(depth):
            inc = in_channels if i == 0 else ch[i - 1]
            self.enc.append(ConvBlock(inc, ch[i], activation=activation))
        self.bottleneck = ConvBlock(ch[depth - 1], ch[depth], activation=activation)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth):
            self.up.append(nn.ConvTranspose2d(ch[depth - i], ch[depth - i - 1], 2, stride=2))
            self.dec.append(ConvBlock(ch[depth - i - 1] * 2, ch[depth - i - 1], activation=activation))
        self.final = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for i in range(self.depth):
            h = self.enc[i](h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for i in range(self.depth):
            h = self.up[i](h)
            s = skips[self.depth - 1 - i]
            h2, h3 = min(h.shape[2], s.shape[2]), min(h.shape[3], s.shape[3])
            h = h[:, :, :h2, :h3]
            s = s[:, :, :h2, :h3]
            h = torch.cat([h, s], dim=1)
            h = self.dec[i](h)
        return self.final(h)


class SpectrogramVocalSeparator(nn.Module):
    """Vocals-only: outputs vocals waveform; accompaniment = mix - vocals."""

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: Optional[int] = None,
        unet_base: int = 32,
        unet_depth: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        activation: str = "leaky_relu",
        mask_type: str = "alpha",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_type = mask_type
        unet_out = (out_channels * 2) if mask_type == "alpha" else out_channels
        self.stft = STFT(n_fft, hop_length, self.win_length)
        self.istft = ISTFT(n_fft, hop_length, self.win_length)
        self.unet = UNet2DSpec(
            in_channels=in_channels,
            out_channels=unet_out,
            base_channels=unet_base,
            depth=unet_depth,
            activation=activation,
        )
        self._unet_depth = unet_depth

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int) -> torch.Tensor:
        B, C, Freq, L = x.shape
        f_pad = ((Freq + multiple - 1) // multiple) * multiple - Freq
        l_pad = ((L + multiple - 1) // multiple) * multiple - L
        if f_pad == 0 and l_pad == 0:
            return x
        return F.pad(x, (0, l_pad, 0, f_pad), mode="constant", value=0.0)

    def _unet_forward(self, mag_log: torch.Tensor) -> torch.Tensor:
        B, C, F, L = mag_log.shape
        multiple = 2 ** self._unet_depth
        padded = self._pad_to_multiple(mag_log, multiple)
        logits = self.unet(padded)
        return logits[:, :, :F, :L]

    def forward(
        self,
        mix: torch.Tensor,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        B, C, T = mix.shape
        if length is None:
            length = T
        if self.in_channels == 2 and self.out_channels == 2 and C >= 2:
            mag_log, phases = self._stereo_mag_phase(mix)
            logits = self._unet_forward(mag_log)
            mask = self._mask_from_logits(logits, 2)
            mag = torch.expm1(mag_log)
            vocal_mag_0 = mask[:, 0:1] * mag[:, 0:1]
            vocal_mag_1 = mask[:, 1:2] * mag[:, 1:2]
            v0 = self.istft(torch.polar(vocal_mag_0, phases[:, 0:1]), length=length)
            v1 = self.istft(torch.polar(vocal_mag_1, phases[:, 1:2]), length=length)
            return torch.cat([v0, v1], dim=1)
        vocals_list = []
        for c in range(C):
            m = mix[:, c : c + 1]
            X = self.stft(m)
            mag = torch.abs(X)
            phase = torch.angle(X)
            mag_log = torch.log1p(mag)
            logits = self._unet_forward(mag_log)
            mask = self._mask_from_logits(logits, 1)
            vocal_mag = mask * mag
            vocal_complex = torch.polar(vocal_mag, phase)
            v = self.istft(vocal_complex, length=length)
            vocals_list.append(v)
        return torch.cat(vocals_list, dim=1)

    def _mask_from_logits(self, logits: torch.Tensor, num_channels: int) -> torch.Tensor:
        if self.mask_type == "alpha":
            B, N, F, L = logits.shape
            logits = logits.view(B, num_channels, 2, F, L)
            mask = torch.softmax(logits, dim=2)[:, :, 0:1].squeeze(2)
            return mask
        return torch.sigmoid(logits)

    def _stereo_mag_phase(self, mix: torch.Tensor):
        X0 = self.stft(mix[:, 0:1])
        X1 = self.stft(mix[:, 1:2])
        mag0 = torch.abs(X0)
        mag1 = torch.abs(X1)
        mag_log = torch.cat([torch.log1p(mag0), torch.log1p(mag1)], dim=1)
        phases = torch.cat([torch.angle(X0), torch.angle(X1)], dim=1)
        return mag_log, phases
