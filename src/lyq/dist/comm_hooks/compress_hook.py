import time

import torch
import torch.distributed as dist

from lyq.dist.env import *
from lyq.dist.comm_hooks._comm_metrix import *

__all__ = [
    'fp16_compress_hook',
    'bf16_compress_hook'
]

_commmetrix = CommMetrix()

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

def fp16_compress_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _compress_hook(
        torch.float16,
        process_group,
        bucket
    )

def bf16_compress_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _compress_hook(
        torch.bfloat16,
        process_group,
        bucket
    )