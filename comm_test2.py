import time

import torch
import torch.distributed as dist

from lyq import *
from lyq.dist import *

if __name__ == '__main__':
    with global_env():
        configs = Configs()
        logger = Logger(
            configs,
            is_multirank=True,
            rank=rank(),
            is_master=is_master()
        )

        num = 0.5
        x = torch.empty(
            int(num * 1024 * 1024 * 1024),
            dtype=torch.uint8,
            device=device()
        )
        x_lists = [
            torch.zeros(
                x.size(),
                dtype=x.dtype,
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
        start_time = time.time()
        for i in range(cnt):
            dist.all_gather(
                x_lists,
                x,
                async_op=False
            )
            torch.cuda.synchronize()
            dist.barrier()
        end_time = time.time()

        logger.info(
            f"{num}GiB All-gather {(end_time - start_time) / cnt}"
        )