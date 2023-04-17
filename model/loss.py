import warnings
from collections import defaultdict
from typing import Dict, Final, Iterable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from utils import angle
import lpips


class Stft(nn.Module):
    def __init__(self, n_fft: int, hop: Optional[int] = None, window: Optional[Tensor] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        # Time-domain input shape: [B, *, T]
        t = input.shape[-1]
        sh = input.shape[:-1]
        out = torch.stft(
            input.reshape(-1, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,
            normalized=True,
            return_complex=True,
        )
        out = out.view(*sh, *out.shape[-2:])
        return out



class MultiResSpecLoss(nn.Module):
    gamma: Final[float]
    f: Final[float]
    f_complex: Final[Optional[List[float]]]

    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: Optional[Union[float, Iterable[float]]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict({str(n_fft): Stft(n_fft) for n_fft in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex = None
        elif isinstance(f_complex, Iterable):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [f_complex] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=input.device, dtype=input.dtype)
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss += F.mse_loss(Y_abs, S_abs) * self.f
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S)) * self.f_complex[i]
        return loss

#PerceptualLoss
class PerceptualLoss(nn.Module):
    def __init__(self, n_fft, net='vgg'):
        super().__init__()
        self.stft = Stft(n_fft)
        self.loss = lpips.LPIPS(net=net)
        for param in self.loss.parameters():
            param.requires_grad = False
            
    def _reshape_source(self, x, source):
        B, S, C, Fr, T = x.shape
        x = torch.concat([x[:,source], x[:,source].mean(dim=1).view(B,1,Fr,T)], dim=1)
        return x
    
    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        input_spec = self.stft(inputs)
        target_spec = self.stft(target)
        drums = self.loss(self._reshape_source(input_spec.abs(), 0), self._reshape_source(target_spec.abs(), 0))
        bass = self.loss(self._reshape_source(input_spec.abs(), 1), self._reshape_source(target_spec.abs(), 1))
        other = self.loss(self._reshape_source(input_spec.abs(), 2), self._reshape_source(target_spec.abs(), 2))
        vocals = self.loss(self._reshape_source(input_spec.abs(), 3), self._reshape_source(target_spec.abs(), 3))
        
        return drums.sum(), bass.sum(), other.sum(), vocals.sum()   