import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from contextlib import contextmanager

from torch import Tensor
from typing import *



def norm_solver(name: str, channels: int) -> nn.Module:
    name = name.lower()
    assert name in {"layer", "batch", "none"}
    if name == "layer":
        return nn.LayerNorm(channels)
    elif name == "batch":
        return nn.BatchNorm1d(channels)
    return nn.Identity()

def act_solver(name: str) -> nn.Module:
    name = name.lower()
    assert name in {"relu", "lrelu", "elu", "none"}
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "lrelu":
        return nn.LeakyReLU(inplace=True)
    elif name == "elu":
        return nn.ELU()
    return nn.Identity()

class MLP(nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        bias: bool = True,
        act: str = "relu",
        norm: str = "layer",
    ) -> None:
        super().__init__()
        assert num_layers > 0
        self.num_layers = num_layers

        if out_channels is None:
            out_channels = hidden_channels
        
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()

        last_ch = in_channels
        for i in range(num_layers):
            if i == num_layers - 1:
                self.linears.append(nn.Linear(last_ch, out_channels, bias=bias))
                break
            else:
                self.linears.append(nn.Linear(last_ch, hidden_channels, bias=bias))

            self.norms.append(norm_solver(norm, hidden_channels))
            self.acts.append(act_solver(act))
            last_ch = hidden_channels
        
    def forward(self, x: Tensor):
        for i in range(self.num_layers):
            x = self.linears[i](x)
            if i == self.num_layers - 1:
                break
            x = self.norms[i](x)
            x = self.acts[i](x)
        return x

class Timer:
    def __init__(self, warmup: int = 1) -> None:
        self._warmup = warmup
        self._total_time = 0.0
        self._total_count = 0
        self._event_pairs: List[Tuple[torch.cuda.Event,...]] = []
    
    @property
    def duration(self) -> float:
        torch.cuda.synchronize()
        dur = 0.0
        for e0, e1 in self._event_pairs:
            dur += e0.elapsed_time(e1)
        self._event_pairs.clear()

        self._total_count += 1
        n = self._total_count - self._warmup
        if n > 0:
            self._total_time += dur
            return self._total_time / n
        else:
            return -1.0

    @contextmanager    
    def record(self):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        try:
            e0.record()
            yield
            e1.record()
            self._event_pairs.append((e0, e1))
        finally:
            pass

comm_timer = Timer()
spmm_timer = Timer()
comp_timer = Timer()
