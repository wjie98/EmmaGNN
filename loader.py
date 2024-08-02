import torch

from torch import Tensor
from typing import *

from starrygl.parallel import Route
from torch_sparse import SparseTensor


class EmmaLoader:
    def __init__(self,
        route: Route,
        adj_t: SparseTensor,
        epochs: int,
        splits: int,
    ) -> None:
        self.route = route

        row, col, val = adj_t.coo()
        if val is None:
            val = torch.ones_like(row)

        self.row = row
        self.col = col
        self.val = val

        self.epochs = epochs
        self.splits = splits
    
    def __iter__(self):
        for ep in range(self.epochs):
            yield self.push_sample(ep)
    
    def push_sample(self, ep: int) -> Tuple[SparseTensor, Route]:
        k = ep % self.splits
        if k == 0:
            self.node_splits = torch.randint(
                0, self.splits,
                size=(self.route.dst_len,),
                dtype=torch.int32,
                device=self.val.device,
            )
        mask = (self.node_splits == k)
        
        dst_mask, src_mask, route = self.route.filter(mask)
        edge_mask = src_mask[self.col]

        row = self.row[edge_mask]
        col = self.col[edge_mask]
        val = self.val[edge_mask]
        # act: Tensor = row.unique()

        # remap source node ids
        imap = torch.full(
            (src_mask.numel(),), (2**62-1)*2+1,
            dtype=col.dtype, device=col.device,
        )
        aind = torch.where(src_mask)[0]
        imap[aind] = torch.arange(aind.numel()).type_as(imap)

        route._bw_ind = imap[route._bw_ind]
        route._src_len = aind.numel()

        col = imap[col]
        
        adj_t = SparseTensor(
            row=row, col=col, value=val,
            sparse_sizes=(
                route.dst_len,
                route.src_len,
            )
        )
        return adj_t, route
    