import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

from starrygl.parallel import Route

from torch_sparse import SparseTensor

from .aggr import EmmaSum
from .utils import MLP, norm_solver, act_solver
from .utils import comm_timer, spmm_timer, comp_timer

class EmmaSAGELayer(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduce_net = EmmaSum()
        self.linear = nn.Linear(in_channels * 2, out_channels, bias=bias)
    
    def forward(self, x: Tensor, adj_t: SparseTensor, route: Route, ema: bool = True) -> Tensor:
        _x = x

        with comm_timer.record():
            x = route.apply(x)
        
        with spmm_timer.record():
            x = adj_t @ x

        with torch.no_grad():
            agg_n = adj_t.sum(dim=1)

        if ema:
            x = self.reduce_net.forward(x, agg_n, aggr="mean")
        else:
            with torch.no_grad():
                inv_w = 1.0 / agg_n
                inv_w[agg_n == 0] = 0.0
            x = x * inv_w[:,None]
        x = torch.cat([x, _x], dim=-1)
        x = self.linear(x)
        return x
    
        
class EmmaSAGE(nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        num_linears: int = 0,
        bias: bool = True,
        dropout: float = 0.0,
        act: str = "relu",
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.dropout = dropout

        if out_channels is None:
            out_channels = hidden_channels

        conv_out_channels = out_channels
        if num_linears > 0:
            conv_out_channels = hidden_channels

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()

        last_ch = in_channels
        for i in range(num_layers):
            if i == num_layers - 1:
                self.layers.append(EmmaSAGELayer(last_ch, conv_out_channels, bias=bias))
                break
            else:
                self.layers.append(EmmaSAGELayer(last_ch, hidden_channels, bias=bias))

            self.norms.append(norm_solver(norm, hidden_channels))
            self.acts.append(act_solver(act))
            last_ch = hidden_channels
        
        if num_linears > 0:
            self.norms.append(norm_solver(norm, conv_out_channels))
            self.acts.append(act_solver(act))
            self.linears = MLP(
                in_channels=conv_out_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_linears,
                bias=bias, act=act, norm=norm,
            )
    
    def forward(self, x: Tensor, adj_t: SparseTensor, route: Route, ema: bool = True):
        with comp_timer.record():
            for i in range(self.num_layers):
                if self.dropout != 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                
                x = self.layers[i](x, adj_t, route=route, ema=ema)
                if i == self.num_layers - 1:
                    break

                x = self.norms[i](x)
                x = self.acts[i](x)
            
            if self.num_linears > 0:
                x = self.norms[-1](x)
                x = self.acts[-1](x)

                if self.dropout != 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.linears(x)
            return x
    
    @torch.no_grad()
    def init_buffers(self, adj_t: SparseTensor):
        agg_n = adj_t.sum(dim=1)
        inv_w = 1.0 / agg_n
        inv_w[agg_n == 0.0] = 0.0
        for layer in self.layers:
            layer: EmmaSAGELayer = layer
            his_x = torch.zeros(
                adj_t.sparse_size(0),
                layer.in_channels,
                device=adj_t.device(),
            )
            layer.reduce_net.init_buffers(his_x, inv_w)

    