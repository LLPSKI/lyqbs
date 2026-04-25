import time

import torch
import torch.distributed as dist

from lyq.dist.env import *
from lyq.dist.comm_hooks._comm_metrix import *

__all__ = [
    'noquan_hook',
    'noquan_get_commmetrix',
]

_commmetrix = CommMetrix()

def noquan_get_commmetrix() -> CommMetrix:
    global _commmetrix
    return _commmetrix

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