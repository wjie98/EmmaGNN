import torch

from torch import Tensor
from typing import *

from collections import defaultdict
from contextlib import contextmanager

from starrygl.distributed import DistributedContext

import json
import time
import logging
import argparse
from pathlib import Path

from models.utils import spmm_timer, comm_timer, comp_timer

class Logger:
    def __init__(self,
        args: argparse.Namespace,
    ) -> None:
        self.ctx = DistributedContext.get_default_context()
        if self.disabled: return

        log_dir = Path(args.log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.log_path = log_dir / time.strftime("%Y%m%d-%H%M%S")
        self.log_path.mkdir(parents=True, exist_ok=True)

        self._save_args(args)
        self.epochs = int(args.epochs)

        self._configure_loggers("loss")
        self._configure_loggers("score")
        self._configure_loggers("perf")

        self.best_val_score = 0.0
        self.best_test_score = 0.0
        self.final_score = 0.0

        self.memory_peak = 0.0
    
    @property
    def disabled(self) -> bool:
        return self.ctx.rank != 0
    
    def _save_args(self, args: argparse.Namespace):
        self.args = args
        with (self.log_path / "params.json").open("w") as f:
            json.dump(self.args.__dict__, f, ensure_ascii=False, indent=2)
    
    def _configure_loggers(self, name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            # "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
            "[%(asctime)s][%(levelname)s] %(message)s"
        )

        log_file_path = self.log_path / f"{name}.log"
        handler = logging.FileHandler(log_file_path.__str__())
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        setattr(self, f"logger_{name}", logger)
    
    def _update_score(self,
        val_score: float,
        test_score: float,
    ):
        if val_score >= self.best_val_score:
            self.best_val_score = val_score
            self.final_score = test_score
        if test_score >= self.best_test_score:
            self.best_test_score = test_score
    
    def _update_memory(self):
        torch.cuda.synchronize()
        self.memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    
    def get_logger(self, name: str) -> logging.Logger:
        return getattr(self, f"logger_{name}")

    def log_perf(self, ep: int):
        if self.disabled: return

        logger = self.get_logger("perf")
        self._update_memory()

        spmm_duration = spmm_timer.duration
        comm_duration = comm_timer.duration
        proj_duration = comp_timer.duration - spmm_duration - comm_duration
        
        avg_duration = (time.time() - self.start_time) / (ep + 1)
        avg_duration *= 1000.0 # s -> ms

        logger.info(
            f"epoch: {ep+1}/{self.epochs} "
            f"epoch_time: {avg_duration:.2f}ms "
            f"peak_memory: {self.memory_peak:.2f}MB "
            f"spmm_time: {spmm_duration:.2f}ms "
            f"comm_time: {comm_duration:.2f}ms "
            f"proj_time: {proj_duration:.2f}ms"
        )
    
    def log_score(self, ep: int, val_score: float, test_score: float):
        if self.disabled: return
        
        logger = self.get_logger("score")
        self._update_score(val_score, test_score)
        
        logger.info(
            f"epoch: {ep+1}/{self.epochs} "
            f"val: {val_score:.2f}/{self.best_val_score:.2f} "
            f"test: {test_score:.2f}/{self.best_test_score:.2f} "
            f"final: {self.final_score:.2f}"
        )
    
    def log_loss(self, ep: int, loss: float):
        if self.disabled: return

        logger = self.get_logger("loss")
        logger.info(
            f"epoch: {ep+1}/{self.epochs} "
            f"loss: {loss:.6f}"
        )
