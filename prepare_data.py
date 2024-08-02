import torch

import argparse

from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_sum

from starrygl.data import GraphData
from pathlib import Path

import logging
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="~/datasets")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["reddit", "yelp", "ogbn-products", "ogbn-papers100M"])
parser.add_argument("--target-root", type=str, default="~/datasets/EmmaGNN")
parser.add_argument("--num_parts", type=int, required=True)
parser.add_argument("--algorithm", type=str, default="pyg-metis", choices=["pyg-metis", "random"])

def load_pt(root: str, name: str) -> GraphData:
    root = Path(root).expanduser().resolve() / f"{name}.pt"
    data: Data = torch.load(root.__str__())
    data.edge_index, _ = add_remaining_self_loops(data.edge_index)
    return GraphData.from_pyg_data(data)

def load_ogb(root: str, name: str) -> GraphData:
    root = Path(root).expanduser().resolve().__str__()
    dataset = PygNodePropPredDataset(name, root=root)

    data = dataset[0]
    assert isinstance(data, Data)

    split_idx = dataset.get_idx_split()
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx["train"], True)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx["valid"], True)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx["test"], True)
    data.edge_index, _ = add_remaining_self_loops(data.edge_index)

    data.y = data.y.squeeze()
    return GraphData.from_pyg_data(data)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    if args.dataset in ["reddit", "yelp"]:
        g = load_pt(args.root, args.dataset)
    elif args.dataset.startswith("ogbn-"):
        g = load_ogb(args.root, args.dataset)
    
    y = g.node()["y"]
    if y.dim() == 1:
        num_classes = y.max().item() + 1
    else:
        assert y.dim() == 2
        num_classes = y.size(1)

    g.meta()["num_classes"] = num_classes
    g.meta()["num_nodes"] = g.node().num_nodes
    g.meta()["num_edges"] = g.edge().num_edges
    g.meta()["partition"] = str(args.algorithm)
    
    node_weight = [
        scatter_sum(
            torch.ones(g.edge().num_edges, dtype=torch.long),
            g.edge_index()[1],
            dim=0, dim_size=g.node().num_nodes,
        )
    ]

    for t in ["train", "val", "test"]:
        mask = g.node()[f"{t}_mask"]
        node_weight.append(mask.long())

    g.node()["node_weight"] = torch.vstack(node_weight).t().contiguous()

    g.save_partition(
        f"{args.target_root}/{args.dataset}",
        args.num_parts,
        node_weight="node_weight",
        ignore_node_attrs=["node_weight"],
        algorithm=args.algorithm,
    )

