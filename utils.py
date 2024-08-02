import torch
import torch.nn as nn
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed import DistributedContext
from sklearn.metrics import f1_score


def all_reduce_score(logits: Tensor, y_true: Tensor, mask: Tensor) -> float:
    ctx = DistributedContext.get_default_context()
    if y_true.dim() == 1:
        y_pred = logits.argmax(dim=-1)
    else:
        y_pred = logits > 0
    
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    y_preds = [None] * ctx.world_size
    dist.all_gather_object(y_preds, y_pred)

    y_trues = [None] * ctx.world_size
    dist.all_gather_object(y_trues, y_true)

    y_pred = torch.cat([t.cpu() for t in y_preds], dim=0)
    y_true = torch.cat([t.cpu() for t in y_trues], dim=0)
    if y_true.dim() == 1:
        return ((y_pred == y_true).sum() / y_true.size(0)).item()
    return f1_score(y_true.cpu(), y_pred.cpu(), average="micro")

    # all_sizes = torch.zeros(ctx.world_size, dtype=torch.long, device=ctx.device)
    # all_sizes[ctx.rank] = y_pred.size(0)
    # dist.reduce(all_sizes, dst=0, op=dist.ReduceOp.SUM)
    # if ctx.rank == 0:
    #     all_sizes = all_sizes.tolist()

    #     all_preds = [y_pred]
    #     all_trues = [y_true]
    #     for i in range(1, ctx.world_size):
    #         s = all_sizes[i]
            
    #         t = torch.empty(
    #             s, *y_pred.shape[1:],
    #             dtype=y_pred.dtype,
    #             device=y_pred.device,
    #         )
            
    #         all_preds.append(
    #             dist.irecv(t, src=i)
    #         )

    #         t = torch.empty(
    #             s, *y_true.shape[1:],
    #             dtype=y_true.dtype,
    #             device=y_true.device,
    #         )

    #         all_trues.append(
    #             dist.irecv(t, src=i)
    #         )
        
    #     for i in range(1, ctx.world_size):
    #         all_preds[i].wait()
    #         all_preds[i] = all_preds[i].result()[0]

    #         all_trues[i].wait()
    #         all_trues[i] = all_trues[i].result()[0]

    #     y_pred = torch.cat(all_preds, dim=0)
    #     y_true = torch.cat(all_trues, dim=0)
    #     if y_true.dim() == 1:
    #         return ((y_pred == y_true).sum() / y_true.size(0)).item()
    #     return f1_score(y_true.cpu(), y_pred.cpu(), average="micro")
    # else:
    #     dist.isend(y_pred, dst=0)
    #     dist.isend(y_true, dst=0)
    #     return -1
    