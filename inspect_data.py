import torch

from torch import Tensor

import argparse
from starrygl.data import GraphData
from pathlib import Path

import logging
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", type=str, required=True)
parser.add_argument("--dataset-name", type=str, required=True,
                    choices=["reddit", "yelp", "ogbn-products", "ogbn-papers100M"])
parser.add_argument("--num-parts", type=int, required=True)
parser.add_argument("--algorithm", type=str, default="pyg-metis", choices=["metis", "pyg-metis", "random"])


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    for i in range(args.num_parts):
        g = GraphData.load_partition(
            f"{args.dataset_root}/{args.dataset_name}",
            part_id=i, num_parts=args.num_parts,
            algorithm=args.algorithm
        )

        print(g.node("dst")["y"].nan_to_num(-1).unique())
        # print(g.node("dst")["y"].nan_to_num(0).count_nonzero() / g.node("dst")["y"].numel())

        print(f"{i+1}/{args.num_parts} node:")
        for type in g.node_types():
            print(f"  type: {type}")
            print(f"    num_nodes: {g.node(type).num_nodes}")
            for key in g.node(type).keys():
                val = g.node(type)[key]
                print(f"    {key}: {val.size()} {val.dtype}")

        print(f"{i+1}/{args.num_parts} edge:")
        for type in g.edge_types():
            print(f"  type: {type}")
            print(f"    num_edges: {g.edge(type).num_edges}")
            for key in g.edge(type).keys():
                val = g.edge(type)[key]
                print(f"    {key}: {val.size()} {val.dtype}")
        
        print(f"{i+1}/{args.num_parts} meta:")
        for key in g.meta().keys():
            val = g.meta()[key]
            if isinstance(val, Tensor):
                val = f"{val.size()} {val.dtype}"
            else:
                val = str(val).splitlines()[0]
            print(f"  {key}: {val}")
