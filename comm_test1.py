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

        num = 2
        x = torch.randn(
            num * 256 * 1024 * 1024,
            dtype=torch.float32,
            device=device()
        )

        # 预热
        dist.all_reduce(
            x,
            async_op=False
        )
        torch.cuda.synchronize()
        dist.barrier()

        cnt = 50
        start = time.time()
        for i in range(cnt):
            dist.all_reduce(
                x,
                async_op=False
            )
            torch.cuda.synchronize()
            dist.barrier()
        end = time.time()

        logger.info(
            f"{num}GiB All-reduce {((end - start) / cnt)}"
        )