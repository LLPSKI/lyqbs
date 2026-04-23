import torch
import torch.distributed as dist
from torch.distributed import (
    ReduceOp,
)
from torch.profiler import (
    record_function,
)
import time

from .env import *

__all__ = [
    "noquan_hook",
    "fp16_compress_hook",
    "bf16_compress_hook",
    "_commmetrix"
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
    
_commmetrix = CommMetrix()

@record_function("noquan_hook")
def noquan_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    assert process_group is None, "noquan_hook默认使用全局进程组，不需要指定！"
    
    process_group = dist.group.WORLD
    big_tensor = bucket.buffer()

    comp_start = time.time()

    big_tensor.div_(process_group.size())

    comp_end = time.time()

    def dethen(fut):
        comm_end = time.time()

        global _commmetrix
        _commmetrix.update(
            comp_end - comp_start,
            comm_end - comm_start,
            comm_bytes
        )

        return fut.value()[0]

    comm_bytes = 2 * ((world_size() - 1) / world_size()) * (
        big_tensor.numel() * big_tensor.element_size()
    )

    comm_start = time.time()
    return dist.all_reduce(
        big_tensor,
        group=process_group,
        async_op=True
    ).get_future().then(dethen)

def _compress_hook(
    dtype: torch.dtype,
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    assert process_group is None, "noquan_hook默认使用全局进程组，不需要指定！"
    process_group = dist.group.WORLD

    big_tensor = bucket.buffer()
    compressed_big_tensor = big_tensor.to(dtype).div_(process_group.size())

    def decompress(fut):
        value = fut.value()[0]
        big_tensor.copy_(value)
        return big_tensor

    return dist.all_reduce(
        compressed_big_tensor,
        group=process_group,
        async_op=True
    ).get_future().then(decompress)

@record_function("fp16_compress_hook")
def fp16_compress_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _compress_hook(
        torch.float16,
        process_group,
        bucket
    )

@record_function("bf16_compress_hook")
def bf16_compress_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _compress_hook(
        torch.bfloat16,
        process_group,
        bucket
    )

class _LyqGradBucket:
    """
    内部测试梯度桶
    GranBucket是torch的内部类，无法直接使用
    """
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
    
    def buffer(self):
        return self._tensor