import time

import torch
import torch.distributed as dist

from lyq.dist.env import *
from lyq.dist.comm_hooks._comm_metrix import *

__all__ = [
    's1e4m3_104_quan_hook',
    's1e4m3_104_quan_get_commmetrix',
]

_commmetrix_s1e4m3_104 = CommMetrix()

def s1e4m3_104_quan_get_commmetrix() -> CommMetrix:
    global _commmetrix_s1e4m3_104
    return _commmetrix_s1e4m3_104

def _s1exmy_base(
    t: torch.Tensor,
    x: int = 4,
    y: int = 3,
    base: int = 104
) -> torch.Tensor:
    """
    对t张量中每一个元素取出符号位1+(阶码-base)后x位+尾数前y位  
    返回相同形状的但元素类型为torch.uint8的张量对象  
    !!!要求：  
    1. t.dtype == torch.float32  
    2. (x >= 0 and x < 8) and (y >= 0 and y < 8) and (x + y) == 7
    3. base >=0 and base < 127
    """
    out = t.view(torch.int32)

    buffer = torch.empty(
        out.size(),
        dtype=out.dtype,
        device=out.device
    )

    mask = torch.empty(
        out.size(),
        dtype=torch.bool,
        device=out.device
    )

    result = torch.zeros(
        out.size(),
        dtype=torch.uint8,
        device=out.device
    )

    # 符号位
    torch.bitwise_right_shift(
        out,
        31,
        out=buffer
    )
    buffer.bitwise_and_(0x1) # 理论来说在右移时高位补0，这里不再需要取最后1位
    buffer.bitwise_left_shift_(7)
    torch.bitwise_or(
        result,
        buffer,
        out=result
    )

    # 阶码位
    torch.bitwise_right_shift(
        out,
        23,
        out=buffer
    )
    buffer.bitwise_and_(0xFF) # 应该不需要
    buffer.sub_(base)
    mask_uint8 = mask.view(dtype=torch.uint8)
    mask_uint8.copy_(buffer)
    torch.ge(
        mask_uint8,
        (1 << x),
        out=mask
    )
    buffer.bitwise_and_((1 << x) - 1)
    buffer.bitwise_left_shift_(7 - x)
    torch.bitwise_or(
        result,
        buffer,
        out=result
    )

    # 尾数位
    torch.bitwise_right_shift(
        out,
        23 - y,
        out=buffer
    )
    buffer.bitwise_and_((1 << y) - 1)
    torch.bitwise_or(
        result,
        buffer,
        out=result
    )

    # 掩码
    result.masked_fill_(mask, 0)

    return result

def _s1exmy_base_decode_and_sum(
    ts: list[torch.Tensor],
    scalar: list[torch.Tensor],
    x: int = 4,
    y: int = 3,
    base: int = 104
) -> torch.Tensor:
    """
    对ts列表中的所有张量中每一个uint8元素，其构成如下：  
    sign exponent mantissa x+y
    o    ooo(x)   ooo(y)   7
    返回相同形状的但元素类型为torch.float32的张量对象  
    !!!要求：  
    1. t.dtype == torch.uint8  
    2. (x >= 0 and x < 8) and (y >= 0 and y < 8) and (x + y) == 7
    3. base >=0 and base < 127
    """
    result = torch.zeros(
        ts[0].size(),
        dtype=torch.float32,
        device=ts[0].device
    )

    buffer_uint8 = torch.empty(
        ts[0].size(),
        dtype=ts[0].dtype,
        device=ts[0].device
    )

    buffer_int32 = torch.empty(
        ts[0].size(),
        dtype=torch.int32,
        device=ts[0].device
    )

    for t, s in zip(ts, scalar):
        # buffer_int32 每次进入需要清零
        buffer_int32.zero_()

        # 符号位
        torch.bitwise_right_shift(
            t,
            7,
            out=buffer_uint8
        )
        sign = buffer_uint8.to(dtype=torch.int32)
        sign.bitwise_left_shift_(31)
        torch.bitwise_or(
            buffer_int32,
            sign,
            out=buffer_int32
        )
        del sign

        # 指数位
        torch.bitwise_right_shift(
            t,
            (7 - x),
            out=buffer_uint8
        )
        exponent = buffer_uint8.to(dtype=torch.int32)
        exponent.bitwise_and_((1 << x) - 1)
        exponent.add_(base)
        exponent.bitwise_left_shift_(23)
        torch.bitwise_or(
            buffer_int32,
            exponent,
            out=buffer_int32
        )
        del exponent

        # 尾数位
        mantissa = t.to(dtype=torch.int32) # 本身就在最末尾，不需要再进行移位操作
        mantissa.bitwise_and_((1 << y) - 1)
        mantissa.bitwise_left_shift_(23 - y)
        torch.bitwise_or(
            buffer_int32,
            mantissa,
            out=buffer_int32
        )
        del mantissa
        
        # 掩码
        buffer_bool = buffer_uint8.view(dtype=torch.bool)
        torch.eq(
            t,
            0,
            out=buffer_bool
        )
        buffer_int32.masked_fill_(buffer_bool, 0)

        # 放大
        buffer_float32 = buffer_int32.view(dtype=torch.float32)
        buffer_float32.mul_(s)

        # 求和
        torch.add(
            result,
            buffer_float32,
            out=result
        )
    
    return result

def _s1exmy_base_quan_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket, # type: ignore
    x: int,
    y: int,
    base: int
) -> torch.futures.Future[torch.Tensor]:
    assert process_group is None, "noquan_hook默认使用全局进程组，不需要指定！"
    
    process_group = dist.group.WORLD
    big_tensor = bucket.buffer()

    comp1_start = time.time()
    # 后续会覆盖掉，原地操作即可，节省中间显存
    big_tensor.div_(process_group.size())

    grad_norm = torch.linalg.vector_norm(big_tensor)
    big_tensor.div_(grad_norm)

    s1exmy_base = _s1exmy_base(
        big_tensor,
        x,
        y,
        base
    )

    grad_norm_lists = [
       torch.zeros(
            1,
            dtype=torch.float32,
            device=s1exmy_base.device
        ) for _ in range(process_group.size())
    ]

    s1exmy_base_lists = [
        torch.zeros(
            s1exmy_base.size(),
            dtype=s1exmy_base.dtype,
            device=s1exmy_base.device
        ) for _ in range(process_group.size())
    ]
    comp1_end = time.time()

    # 通信量计算
    comm_bytes = 4 + s1exmy_base.numel()

    comm_start = time.time()
    dist.all_gather(
        grad_norm_lists,
        grad_norm,
        process_group,
        async_op=False
    )
    dist.all_gather(
        s1exmy_base_lists,
        s1exmy_base,
        process_group,
        async_op=False
    )
    # torch.cuda.synchronize(s1exmy_base.device)
    del grad_norm, s1exmy_base
    comm_end = time.time()


    comp2_start = time.time()
    result = _s1exmy_base_decode_and_sum(
        s1exmy_base_lists,
        grad_norm_lists,
        x,
        y,
        base
    )
    big_tensor.copy_(result)
    comp2_end = time.time()

    global _commmetrix_s1e4m3_104
    _commmetrix_s1e4m3_104.update(
        (comp1_end - comp1_start) + (comp2_end - comp2_start),
        comm_end - comm_start,
        comm_bytes
    )

    fut = torch.futures.Future[torch.Tensor]()
    fut.set_result(big_tensor)

    return fut


def s1e4m3_104_quan_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _s1exmy_base_quan_hook(
        process_group,
        bucket,
        4,
        3, 
        104
    )