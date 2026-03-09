"""U-Net spectrogram multi-stem separation (vocals, drums, bass, other)."""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stft_istft import STFT, ISTFT
from .unet_spec import UNet2DSpec

STEM_NAMES: List[str] = ["vocals", "drums", "bass", "other"]


class SpectrogramStemsSeparator(nn.Module):
    """Spectrogram separator that outputs 4 stems (vocals, drums, bass, other) via softmax masks."""

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: Optional[int] = None,
        unet_base: int = 32,
        unet_depth: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        n_sources: int = 4,
        activation: str = "leaky_relu",
        mask_type: str = "alpha",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_sources = n_sources
        self.mask_type = mask_type
        unet_out = out_channels * n_sources
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

    def _mask_from_logits(self, logits: torch.Tensor, num_channels: int) -> torch.Tensor:
        B, N, F, L = logits.shape
        logits = logits.view(B, num_channels, self.n_sources, F, L)
        return torch.softmax(logits, dim=2)

    def _stereo_mag_phase(self, mix: torch.Tensor):
        X0 = self.stft(mix[:, 0:1])
        X1 = self.stft(mix[:, 1:2])
        mag0 = torch.abs(X0)
        mag1 = torch.abs(X1)
        mag_log = torch.cat([torch.log1p(mag0), torch.log1p(mag1)], dim=1)
        phases = torch.cat([torch.angle(X0), torch.angle(X1)], dim=1)
        return mag_log, phases

    def forward(
        self,
        mix: torch.Tensor,
        length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        B, C, T = mix.shape
        if length is None:
            length = T
        names = STEM_NAMES[: self.n_sources]

        if self.in_channels == 2 and self.out_channels == 2 and C >= 2:
            mag_log, phases = self._stereo_mag_phase(mix)
            logits = self._unet_forward(mag_log)
            masks = self._mask_from_logits(logits, 2)
            mag = torch.expm1(mag_log)
            out = {}
            for i, name in enumerate(names):
                m0 = masks[:, 0:1, i] * mag[:, 0:1]
                m1 = masks[:, 1:2, i] * mag[:, 1:2]
                c0 = self.istft(torch.polar(m0, phases[:, 0:1]), length=length)
                c1 = self.istft(torch.polar(m1, phases[:, 1:2]), length=length)
                out[name] = torch.cat([c0, c1], dim=1)
            return out

        out: Dict[str, torch.Tensor] = {}
        for c in range(C):
            m = mix[:, c : c + 1]
            X = self.stft(m)
            mag = torch.abs(X)
            phase = torch.angle(X)
            mag_log = torch.log1p(mag)
            logits = self._unet_forward(mag_log)
            masks = self._mask_from_logits(logits, 1)
            for src_i, name in enumerate(names):
                stem_mag = (masks[:, 0, src_i] * mag.squeeze(1)).unsqueeze(1)
                stem_complex = torch.polar(stem_mag, phase)
                wav = self.istft(stem_complex, length=length)
                if name not in out:
                    out[name] = []
                out[name].append(wav)
        return {name: torch.cat(out[name], dim=1) for name in names}
