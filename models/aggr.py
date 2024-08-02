import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch import Tensor
from typing import *

from torch_scatter import segment_csr, gather_csr
from torch_sparse import SparseTensor


__all__ = [
    "EmmaAttention",
    "EmmaSum",
]


class EmmaAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor, max_a: Tensor, agg_n: Tensor):
        
        his_x = self.get_buffer("his_x")
        his_m = self.get_buffer("his_m")
        inv_w = self.get_buffer("inv_w")
        
        x = EmmaAttentionFunction.apply(x, max_a, his_x, his_m, agg_n, inv_w, self.training)
        return x
    
    def init_buffers(self, his_x: Tensor, his_m: Tensor, inv_w: Tensor):
        self.register_buffer("his_x", his_x, persistent=False)
        self.register_buffer("his_m", his_m, persistent=False)
        self.register_buffer("inv_w", inv_w, persistent=False)
    
    @staticmethod
    def softmax_gat(
        src_a: Tensor,
        dst_a: Tensor,
        adj_t: SparseTensor,
        negative_slope: float = 0.01,
    ) -> Tuple[SparseTensor, Tensor]:
        assert src_a.dim() in {1, 2}
        assert src_a.dim() == dst_a.dim()

        ptr, ind, val = adj_t.csr()
        
        a = src_a[ind] + gather_csr(dst_a, ptr)
        a = F.leaky_relu(a, negative_slope=negative_slope)

        with torch.no_grad():
            max_a = torch.full_like(dst_a, -torch.inf)
            max_a = segment_csr(a, ptr, reduce="max", out=max_a)
        exp_a = torch.exp(a - gather_csr(max_a, ptr))

        if val is not None:
            assert val.dim() == 1
            if exp_a.dim() == 1:
                exp_a = exp_a * val
            else:
                exp_a = exp_a * val.unsqueeze(-1)

        sum_exp_a = segment_csr(exp_a, ptr, reduce="sum")
        exp_a = exp_a / gather_csr(sum_exp_a, ptr)
        with torch.no_grad():
            max_a.add_(sum_exp_a.log())

        adj_t = SparseTensor(rowptr=ptr, col=ind, value=exp_a)
        return adj_t, max_a
    
    @staticmethod
    def apply_gat(
        x: Tensor,
        src_a: Tensor,
        dst_a: Tensor,
        adj_t: SparseTensor,
        negative_slope: float = 0.01,
    ) -> Tuple[Tensor, Tensor]:
        adj_t, max_a = EmmaAttention.softmax_gat(
            src_a=src_a, dst_a=dst_a,
            adj_t=adj_t, negative_slope=negative_slope,
        )

        ptr, ind, val = adj_t.csr()
        if val.dim() == 1:
            assert x.dim() == 2
            x = adj_t @ x
        elif val.dim() == 2:
            assert x.dim() == 3
            assert x.size(1) == val.size(1)
            xs = []
            for i in range(x.size(1)):
                xs.append(
                    SparseTensor(
                        rowptr=ptr, col=ind, value=val[:,i],
                    ) @ x[:,i,:]
                )
            x = torch.cat(xs, dim=1).view(-1, *x.shape[1:])

        return x, max_a

class EmmaAttentionFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        x: Tensor,
        max_a: Tensor,
        his_x: Tensor,
        his_m: Tensor,
        agg_n: Tensor,
        inv_w: Tensor,
        training: bool,
    ):
        assert x.dim() in {2, 3}
        assert x.dim() == his_x.dim()
        assert max_a.dim() == his_m.dim()

        if training:
            beta = (1.0 - inv_w * agg_n).clamp_(0.0, 1.0)
            if x.dim() == 2:
                assert max_a.dim() == 1
            elif x.dim() == 3:
                assert max_a.dim() == 2
                beta = beta.unsqueeze_(-1)

            max_m = torch.max(max_a, his_m)

            p = (his_m - max_m).nan_to_num_(-torch.inf).exp_().mul_(beta)
            q = (max_a - max_m).nan_to_num_(-torch.inf).exp_()

            t = (p + q).clamp_min_(1.0)
            p.div_(t).unsqueeze_(-1)
            q.div_(t).unsqueeze_(-1)

            his_x.mul_(p).add_(x * q)
            his_m.copy_(max_m).add_(t.log_())
            ctx.save_for_backward(q)
        else:
            his_x.copy_(x)
            his_m.copy_(max_a)
        return his_x
    
    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ):
        q, = ctx.saved_tensors
        return grad * q, None, None, None, None, None, None

class EmmaSum(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor, agg_n: Tensor, aggr: str = "sum"):
        assert aggr in {"sum", "mean"}

        his_x = self.get_buffer("his_x")
        inv_w = self.get_buffer("inv_w")
        x = EmmaSumFunction.apply(x, his_x, agg_n, inv_w, self.training)

        if aggr == "mean":
            x = x * inv_w[:,None]
        return x
    
    def init_buffers(self, his_x: Tensor, inv_w: Tensor):
        self.register_buffer("his_x", his_x, persistent=False)
        self.register_buffer("inv_w", inv_w, persistent=False)

class EmmaSumFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        x: Tensor,
        his_x: Tensor,
        agg_n: Tensor,
        inv_w: Tensor,
        training: bool
    ):
        assert x.dim() == 2
        assert his_x.dim() == x.dim()
        if training:
            beta = (1.0 - inv_w * agg_n) \
                .clamp_(0.0, 1.0).unsqueeze_(-1)
            his_x.mul_(beta).add_(x)
        else:
            his_x.copy_(x)
        return his_x
    
    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ):
        return grad, None, None, None, None