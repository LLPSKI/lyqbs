import time

import torch
import torch.distributed as dist

from lyq import *
from lyq.dist import *
from lyq.dist.comm_hooks.s1exmy_base_quan_hook import (
    _s1exmy_base,
    _s1exmy_base_decode_and_sum
)

if __name__ == '__main__':
    with global_env():
        configs = Configs()
        logger = Logger(
            configs,
            is_multirank=True,
            rank=rank(),
            is_master=is_master()
        )

        num = 2
        x = torch.randn(
            num * 256 * 1024 * 1024,
            dtype=torch.float32,
            device=device()
        )
        x_lists = [
            torch.zeros(
                x.size(),
                dtype=torch.uint8,
                device=x.device
            ) for _ in range(world_size())
        ]
        grad_norm_lists = [
            torch.tensor(
                1.0,
                dtype=torch.float32,
                device=x.device
            ) for _ in range(world_size())
        ]

        # 预热
        dist.all_reduce(
            x,
            async_op=False
        )
        torch.cuda.synchronize()
        dist.barrier()

        cnt = 50
        comp_start = time.time()
        for i in range(cnt):
            big_tensor = x.clone()
            grad_norm = torch.linalg.vector_norm(big_tensor)
            big_tensor.div_(grad_norm)
            s1exmy_base = _s1exmy_base(
                big_tensor,
                4,
                3,
                104
            )
            result = _s1exmy_base_decode_and_sum(
                x_lists,
                grad_norm_lists,
                4,
                3,
                104
            )
            big_tensor.copy_(result)
        comp_end = time.time()

        logger.info(
            f"comp {(comp_end - comp_start) / cnt}"
        )