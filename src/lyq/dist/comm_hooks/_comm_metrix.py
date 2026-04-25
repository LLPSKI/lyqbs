import torch
import torch.distributed as dist
from torch.distributed import (
    ReduceOp
)

from lyq.dist.env import *

__all__ = [
    'CommMetrix'
]

class CommMetrix:
    def __init__(self):
        self.comm_metrix = torch.tensor(
            [0.0, 0.0, 0.0, 0],
            dtype=torch.float,
            device=device()
        )
    
    def update(
        self,
        comp_time: float,
        comm_time: float,
        comm_bytes: float
    ) -> None:
        self.comm_metrix[0] = comp_time + comm_time
        self.comm_metrix[1] = comp_time
        self.comm_metrix[2] = comm_time
        self.comm_metrix[3] = comm_bytes
    
    def all_reduce(
        self,
    ) -> None:
        dist.all_reduce(
            self.comm_metrix,
            ReduceOp.AVG,
            async_op=True
        )
    
    @property
    def total_time(self) -> float:
        return self.comm_metrix[0].item()

    @property
    def comp_time(self) -> float:
        return self.comm_metrix[1].item()

    @property
    def comm_time(self) -> float:
        return self.comm_metrix[2].item()
    
    @property
    def comm_bytes(self) -> int:
        return int(self.comm_metrix[3].item())