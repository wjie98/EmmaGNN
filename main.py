import torch
import torch.nn as nn
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed import DistributedContext
from starrygl.data import GraphData
from starrygl.parallel import Route

from models.sage import EmmaSAGE
from models.attn import EmmaGAT
from models.utils import comm_timer, spmm_timer, comp_timer
from utils import all_reduce_score

from loader import EmmaLoader
from logger import Logger

from tqdm import tqdm

from torch_sparse import SparseTensor
from sklearn.metrics import f1_score

import os
import argparse

import logging
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", type=str, required=True)
parser.add_argument("--dataset-name", type=str, required=True)
parser.add_argument("--model", type=str, default="sage", choices=["sage", "gat"])
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--num-linears", type=int, default=0)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--splits", type=int, default=10)
parser.add_argument("--interval", type=int, default=0)
parser.add_argument("--emma", action="store_true")
parser.add_argument("--log-dir", type=str, default="logs")
parser.add_argument("--use-fp16", action="store_true")
parser.add_argument("--trim-layers", action="store_true")

def get_adj_t(g: GraphData) -> SparseTensor:
    edge_index = g.edge_index()
    edge_attr = torch.ones(
        g.edge().num_edges,
        device=edge_index.device,
    )
    adj_t = SparseTensor.from_edge_index(
        edge_index, edge_attr,
        sparse_sizes=(
            g.node("src").num_nodes,
            g.node("dst").num_nodes,
        )
    ).t().coalesce(reduce="sum")
    return adj_t

def find_module(net: nn.Module, cls):
    if isinstance(net, cls):
        return net
    for m in net.modules():
        if isinstance(m, cls):
            return m

def get_loss_scale(mask: Tensor) -> float:
    s = mask.count_nonzero()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    return 1.0 / s.item()

def all_reduce_num_classes(y: Tensor) -> int:
    y = y.nan_to_num(-1).long()
    if y.dim() == 1:
        num_classes = y.max() + 1
        dist.all_reduce(num_classes, op=dist.ReduceOp.MAX)
        return int(num_classes.item())
    else:
        assert y.dim() == 2
        num_classes = y.size(1)
        return int(num_classes)

def trim_by_layers(g: GraphData, num_layers: int):
    from torch_scatter import scatter_max
    
    route = g.to_route()
    edge_index = g.edge_index()

    dst_mask = g.node("dst")["train_mask"] | g.node("dst")["val_mask"] | g.node("dst")["test_mask"]
    for _ in range(num_layers):
        edge_mask = dst_mask[edge_index[1]].long()
        src_mask, _ = scatter_max(edge_mask, edge_index[0], dim=0, dim_size=g.node("src").num_nodes)
        dst_mask |= (route.bw_tensor(src_mask) != 0)
    src_mask = route.fw_tensor(dst_mask.long()) != 0

    edge_mask = src_mask[edge_index[0]] & dst_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    src_ind = torch.where(src_mask)[0]
    imap = torch.full((src_mask.numel(),), (2**62-1)*2+1, dtype=torch.long, device=src_mask.device)
    imap[src_ind] = torch.arange(src_ind.numel(), dtype=torch.long, device=src_mask.device)
    src = imap[edge_index[0]]

    dst_ind = torch.where(dst_mask)[0]
    imap = torch.full((dst_mask.numel(),), (2**62-1)*2+1, dtype=torch.long, device=dst_mask.device)
    imap[dst_ind] = torch.arange(dst_ind.numel(), dtype=torch.long, device=dst_mask.device)
    dst = imap[edge_index[1]]

    edge_index = torch.vstack([src, dst])
    raw_src_ids = g.node("src")["raw_ids"][src_mask]
    raw_dst_ids = g.node("dst")["raw_ids"][dst_mask]

    sg = GraphData.from_bipartite(
        edge_index=edge_index,
        raw_src_ids=raw_src_ids,
        raw_dst_ids=raw_dst_ids,
    )
    sg.node("dst")["x"] = g.node("dst")["x"][dst_mask]
    sg.node("dst")["y"] = g.node("dst")["y"][dst_mask]
    sg.node("dst")["train_mask"] = g.node("dst")["train_mask"][dst_mask]
    sg.node("dst")["val_mask"] = g.node("dst")["val_mask"][dst_mask]
    sg.node("dst")["test_mask"] = g.node("dst")["test_mask"][dst_mask]
    return sg

def print_memory(s):
    ctx = DistributedContext.get_default_context()
    torch.cuda.synchronize()
    ctx.sync_print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024))

def train_once(
    net: Union[EmmaSAGE, EmmaGAT],
    adj_t: SparseTensor,
    route: Route,
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: Union[nn.CrossEntropyLoss, nn.BCEWithLogitsLoss],
    loss_scale: float,
    ema: bool,
) -> Tensor:
    net.train()
    if optimizer is not None:
        optimizer.zero_grad()

    logits = net.forward(x, adj_t, route, ema=ema)
    loss: Tensor = criterion(logits[mask], y[mask]) * loss_scale
    loss.backward()

    if optimizer is not None:
        optimizer.step()
    return loss.detach()

@torch.no_grad()
def eval_once(
    net: Union[EmmaSAGE, EmmaGAT],
    adj_t: SparseTensor,
    route: Route,
    x: Tensor,
    y: Tensor,
    val_mask: Tensor,
    test_mask: Optional[Tensor],
    ema: bool,
) -> Tensor:
    net.eval()
    logits: Tensor = net.forward(x, adj_t, route, ema=ema)

    val_score = all_reduce_score(logits, y, val_mask)
    if test_mask is not None:
        test_score = all_reduce_score(logits, y, test_mask)
    else:
        test_score = -1
    return val_score * 100.0, test_score * 100.0

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "24"
    args = parser.parse_args()
    ctx = DistributedContext.init(args.backend, use_gpu=True)

    if args.interval <= 0:
        args.interval = args.splits ** 2
    logger = Logger(args)

    g = GraphData.load_partition(
        f"{args.dataset_root}/{args.dataset_name}",
        part_id=ctx.rank, num_parts=ctx.world_size,
        algorithm="pyg-metis",
    ).to(ctx.device)

    if args.use_fp16:
        torch.set_default_dtype(torch.float16)
        g.node("dst")["x"] = g.node("dst")["x"].to(torch.float16)
    
    if args.trim_layers:
        g = trim_by_layers(g, args.num_layers)

    num_features = g.node("dst")["x"].size(-1)
    num_classes = all_reduce_num_classes(g.node("dst")["y"])

    x = g.node("dst")["x"]
    y = g.node("dst")["y"].nan_to_num(num_classes).long()
    train_mask = g.node("dst")["train_mask"]
    val_mask = g.node("dst")["val_mask"]
    test_mask = g.node("dst")["test_mask"]
    loss_scale = get_loss_scale(train_mask)

    if g.node("dst")["y"].dim() == 1:
        criterion = nn.CrossEntropyLoss(reduction="sum")
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
        y = y.float()

    route = g.to_route()
    adj_t = get_adj_t(g)
    del g
    
    if args.model == "sage":
        net = EmmaSAGE(
            in_channels=num_features,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            out_channels=num_classes,
            num_linears=args.num_linears,
            dropout=args.dropout,
        ).to(ctx.device)
    elif args.model == "gat":
        net = EmmaGAT(
            in_channels=num_features,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            out_channels=num_classes,
            num_linears=args.num_linears,
            dropout=args.dropout,
        ).to(ctx.device)
    net = nn.parallel.DistributedDataParallel(net)
    find_module(net, (EmmaSAGE, EmmaGAT)).init_buffers(adj_t)

    if args.use_fp16:
        optimizer = scheduler = None
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.1)

    bar = None
    if args.emma:
        loader = EmmaLoader(route, adj_t, epochs=args.epochs, splits=args.splits)

        for ep, (_adj_t, _route) in enumerate(loader):
            if ep % args.interval == 0:
                if bar is not None:
                    bar.close()
                    bar = None

                val_score, test_score = eval_once(net, adj_t, route, x, y, val_mask, test_mask, ema=True)
                if ctx.rank == 0:
                    logger.log_score(ep, val_score, test_score)

            loss = train_once(net, _adj_t, _route, x, y, train_mask, optimizer, criterion, loss_scale, ema=True).item()
            
            logger.log_loss(ep, loss)
            logger.log_perf(ep)

            if scheduler is not None:
                scheduler.step()

            if bar is None:
                bar = tqdm(total=args.interval, disable=(ctx.rank!=0))
            bar.set_description(f"{ep+1}/{args.epochs} loss: {loss:.6f}")
            bar.update()
    else:
        for ep in range(args.epochs):
            if ep % args.interval == 0:
                if bar is not None:
                    bar.close()
                    bar = None

                val_score, test_score = eval_once(net, adj_t, route, x, y, val_mask, test_mask, ema=False)
                if ctx.rank == 0:
                    logger.log_score(ep, val_score, test_score)

            loss = train_once(net, adj_t, route, x, y, train_mask, optimizer, criterion, loss_scale, ema=False)

            logger.log_loss(ep, loss.item())
            logger.log_perf(ep)

            if scheduler is not None:
                scheduler.step()

            if bar is None:
                bar = tqdm(total=args.interval, disable=(ctx.rank!=0))
            bar.set_description(f"{ep+1}/{args.epochs} loss: {loss.item():.6f}")
            bar.update()

    if bar is not None:
        bar.close()
        bar = None

    val_score, test_score = eval_once(net, adj_t, route, x, y, val_mask, test_mask, ema=False)
    if ctx.rank == 0:
        logger.log_score(ep, val_score, test_score)