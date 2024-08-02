import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as ckpt

from torch import Tensor
from typing import *

from starrygl.parallel import Route

from torch_scatter import segment_csr, gather_csr
import torch_sparse as tsp
from torch_sparse import SparseTensor, cat as spcat
from torch_geometric.nn import GATConv
import torch_geometric.utils as pyg_utils

from .aggr import EmmaAttention
from .utils import MLP, norm_solver, act_solver
from .utils import comm_timer, spmm_timer, comp_timer

class EmmaGATLayer(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        if heads is None:
            self.att_src = nn.Parameter(torch.zeros(out_channels))
            self.att_dst = nn.Parameter(torch.zeros(out_channels))
        else:
            self.att_src = nn.Parameter(torch.zeros(heads, out_channels))
            self.att_dst = nn.Parameter(torch.zeros(heads, out_channels))
            out_channels = out_channels * heads

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.reduce_net = EmmaAttention()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear.reset_parameters()
        if self.heads is None:
            nn.init.xavier_uniform_(self.att_src.data.view(-1, 1))
            nn.init.xavier_uniform_(self.att_dst.data.view(-1, 1))
        else:
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)
        if hasattr(self, "bias"):
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor, adj_t: SparseTensor, route: Route, ema: bool = True) -> Tensor:

        x = self.linear(x)
        if self.heads is not None:
            assert x.size(-1) % self.heads == 0
            x = x.view(x.size(0), self.heads, -1)
        
        src_a = (x * self.att_src).sum(dim=-1)
        dst_a = (x * self.att_dst).sum(dim=-1)

        _x = x

        with comm_timer.record():
            x = route.apply(x)
            src_a = route.apply(src_a)
        
        with spmm_timer.record():
            x, max_a = self.reduce_net.apply_gat(x, src_a, dst_a, adj_t)
        
        with torch.no_grad():
            agg_n = adj_t.sum(dim=1)

        if ema:
            x = self.reduce_net.forward(x, max_a, agg_n)
        x = (x + _x).flatten(start_dim=1)

        if hasattr(self, "bias"):
            x = x + self.bias
        return x
        
class EmmaGAT(nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        num_linears: int = 0,
        bias: bool = True,
        heads: Optional[int] = None,
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
        conv_out_heads = heads
        if num_linears > 0:
            conv_out_channels = hidden_channels
        elif heads is not None:
            conv_out_heads = 1

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()

        last_ch = in_channels
        head_op = 1 if heads is None else heads
        for i in range(num_layers):
            if i == num_layers - 1:
                self.layers.append(EmmaGATLayer(
                    last_ch, conv_out_channels, bias=bias, heads=conv_out_heads))
                break
            else:
                self.layers.append(
                    EmmaGATLayer(last_ch, hidden_channels, bias=bias, heads=heads))

            self.norms.append(norm_solver(norm, hidden_channels * head_op))
            self.acts.append(act_solver(act))
            last_ch = hidden_channels * head_op
        
        if num_linears > 0:
            self.norms.append(norm_solver(norm, conv_out_channels * head_op))
            self.acts.append(act_solver(act))
            self.linears = MLP(
                in_channels=conv_out_channels * head_op,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
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
            layer: EmmaGATLayer = layer
            if layer.heads is None:
                sizes = (adj_t.sparse_size(0), layer.out_channels)
            else:
                sizes = (adj_t.sparse_size(0), layer.heads, layer.out_channels)
            his_x = torch.zeros(*sizes, device=adj_t.device())
            his_m = torch.full(sizes[:-1], -torch.inf, device=adj_t.device())
            layer.reduce_net.init_buffers(his_x, his_m, inv_w)
