import time

import torch
import torch.distributed as dist

from lyq.dist.env import *
from lyq.dist.comm_hooks._comm_metrix import *

__all__ = [
    's1e4m3_104_quan_hook',
    's1e3m4_112_quan_hook',
    's1e2m5_116_quan_hook',
    's1e1m6_118_quan_hook',
    's1e0m7_119_quan_hook',
    's1exmy_base_generator',
    's1exmy_base_commmetrix'
]

_s1exmy_base_commmetrix = CommMetrix()

def s1exmy_base_commmetrix() -> CommMetrix:
    global _s1exmy_base_commmetrix
    return _s1exmy_base_commmetrix

_s1exmy_base_generator = torch.Generator(device()).manual_seed(9527)

def s1exmy_base_generator() -> torch.Generator:
    global _s1exmy_base_generator
    return _s1exmy_base_generator

def _s1exmy_base_sign(
    out: torch.Tensor,
    result: torch.Tensor
) -> None:
    """
    取出符号位  
    将输入张量对象中的最高位符号位取出来放入输出张量对象的最高位中。   
    +1      sign    result  
    0        0         0  
    0        1         1    ->   
    1        0         0    ->  
    1        1         1  

    Args: 
        out: 输入张量对象，元素类型为torch.int32  
        result: 输出张量对象，元素类型为torch.uint8  
    """
    buffer = out < 0
    buffer_1 = result >= 128
    torch.bitwise_xor(
        buffer,
        buffer_1,
        out=buffer
    )
    torch.where(
        buffer,
        result + 128,
        result,
        out=result
    )

def _s1exmy_base_exponent_and_mask(
    out: torch.Tensor,
    result: torch.Tensor,
    x: int = 4,
    base: int = 104,
) -> torch.Tensor:
    """
    取出阶码位  
    将输入张量对象中的阶码位取出来并减去base，再将二进制表示的后x位放入输出张量对象的符号位的后x位。  
    根据输入张量对象中的阶码值是否满足要求：[base, base+(1<<x)-1]，以此生成布尔掩码张量。  

    Args: 
        out: 输入张量对象，元素类型为torch.int32  
        result: 输出张量对象，元素类型为torch.uint8  
        x: 需要取出的阶码位数  
        base: 偏移值  
    
    Returns:
        输出的布尔掩码张量对象，元素类型为torch.bool
    """
    buffer = out >> 23
    buffer.bitwise_and_(0xFF) # 应该不需要
    buffer.sub_(base)

    mask = torch.empty(
        out.size(),
        dtype=torch.bool,
        device=out.device
    )
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

    return mask

def _s1exmy_base_mantissa_and_bernoulli(
    out: torch.Tensor,
    result: torch.Tensor,
    y: int = 3,
) -> torch.Tensor:
    """
    取出尾数位  
    将输入张量对象中的尾数位的前y位放入输出张量对象的后y位。  
    根据输入张量对象中剩余的尾数值除以(2 << (23 - y))得到概率。  

    Args: 
        out: 输入张量对象，元素类型为torch.int32  
        result: 输出张量对象，元素类型为torch.uint8  
        y: 需要取出的尾数位数  
    
    Returns:
        输出的布尔掩码张量对象，元素类型为torch.bool
    """
    buffer = out >> (23 - y)
    buffer.bitwise_and_((1 << y) - 1)
    torch.bitwise_or(
        result,
        buffer,
        out=result
    )
    del buffer
    
    buffer = out & ((1 << (23 - y)) - 1)
    buffer_float32 = buffer.to(dtype=torch.float32)
    buffer_float32.div_((1 << (23 - y)))
    
    return torch.bernoulli(
        buffer_float32,
        generator=_s1exmy_base_generator,
    ).to(dtype=torch.bool)

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

    Args: 
        out: 输入张量对象，元素类型为torch.int32  
        result: 输出张量对象，元素类型为torch.uint8  
        x: 需要取出的阶码位数  
        y: 需要取出的尾数位数  
        base: 偏移值  
    
    Returns:
        输出的编码张量对象，元素类型为torch.uint8
    """
    out = t.view(torch.int32)

    result = torch.zeros(
        out.size(),
        dtype=torch.uint8,
        device=out.device
    )

    mask = _s1exmy_base_exponent_and_mask(out, result, x, base)
    mask_add = _s1exmy_base_mantissa_and_bernoulli(out, result, y)

    # 随机
    result.add_(mask_add)

    # 掩码
    result.masked_fill_(mask, 0)

    _s1exmy_base_sign(out, result)

    return result

def _s1exmy_base_decode_sign(
    t: torch.Tensor,
    result: torch.Tensor,
) -> None:
    """
    解码符号位
    """
    buffer_uint8 = t >> 7
    buffer_int32 = buffer_uint8.to(dtype=torch.int32)
    buffer_int32.bitwise_left_shift_(31)
    torch.bitwise_or(
        result,
        buffer_int32,
        out=result
    )

def _s1exmy_base_decode_exponent(
    t: torch.Tensor,
    result: torch.Tensor,
    x: int = 4,
    base: int = 104
) -> None:
    """
    解码阶码位
    """
    buffer_uint8 = t >> (7 - x)
    buffer_int32 = buffer_uint8.to(dtype=torch.int32)
    buffer_int32.bitwise_and_((1 << x) - 1)
    buffer_int32.add_(base)
    buffer_int32.bitwise_left_shift_(23)
    torch.bitwise_or(
        result,
        buffer_int32,
        out=result
    )

def _s1exmy_base_decode_mantissa(
    t: torch.Tensor,
    result: torch.Tensor,
    y: int = 3
) -> None:
    """
    解码尾数位
    """
    buffer_int32 = t.to(dtype=torch.int32)
    buffer_int32.bitwise_and_((1 << y) - 1)
    buffer_int32.bitwise_left_shift_(23 - y)
    torch.bitwise_or(
        result,
        buffer_int32,
        out=result
    )

def _s1exmy_base_decode_mask(
    t: torch.Tensor,
) -> torch.Tensor:
    """
    得到掩码，等于0x0000_0000或者0x1000_0000
    """
    return (t == 0) | (t == 128)


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

    buffer_int32 = torch.empty(
        ts[0].size(),
        dtype=torch.int32,
        device=ts[0].device
    )

    for t, s in zip(ts, scalar):
        # buffer_int32 每次进入需要清零
        buffer_int32.zero_()

        _s1exmy_base_decode_sign(t, buffer_int32)
        _s1exmy_base_decode_exponent(t, buffer_int32, x, base)
        _s1exmy_base_decode_mantissa(t, buffer_int32, y)
        mask = _s1exmy_base_decode_mask(t)
        buffer_int32.masked_fill_(mask, 0)

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

    global _s1exmy_base_commmetrix
    _s1exmy_base_commmetrix.update(
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

def s1e3m4_112_quan_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _s1exmy_base_quan_hook(
        process_group,
        bucket,
        3,
        4, 
        112
    )

def s1e2m5_116_quan_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _s1exmy_base_quan_hook(
        process_group,
        bucket,
        2,
        5, 
        116
    )

def s1e1m6_118_quan_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _s1exmy_base_quan_hook(
        process_group,
        bucket,
        1,
        6, 
        118
    )

def s1e0m7_119_quan_hook(
    process_group: dist.ProcessGroup, 
    bucket: dist.GradBucket # type: ignore
) -> torch.futures.Future[torch.Tensor]:
    return _s1exmy_base_quan_hook(
        process_group,
        bucket,
        0,
        7, 
        119
    )

if __name__ == '__main__':
    import struct
    from lyq import *
    from lyq.utils.bin import binstr_to_float32
    with global_env():
        configs = Configs()
        logger = Logger(
            configs,
            is_multirank=True,
            rank=rank(),
            is_master=is_master()
        )

        logger.info(
            f'开始测试_s1exmy_base...\n'
        )

        #############################################################################################
        logger.info(
            f"开始测试_s1exmy_base_sign()...\n"
            f"构建测试用例中..."
        )
        x = torch.tensor(
            [
                binstr_to_float32(str('0_01111001_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01111010_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01111011_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01111100_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01111101_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('1_01110011_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('1_01110100_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('1_01110101_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('1_01110110_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('1_01110111_00000000000000000000000').replace('_', '')),
            ],
            dtype=torch.float32,
            device=device()
        )
        out = x.view(dtype=torch.int32)
        result = torch.zeros(
            out.size(),
            dtype=torch.uint8,
            device=out.device
        )

        _s1exmy_base_sign(out, result)
        logger.info(
            f"{result}"
        )
        result.zero_()

        #############################################################################################
        logger.info(
            f"开始测试_s1exmy_base_exponent_and_mask()..."
        )
        x = torch.tensor(
            [
                binstr_to_float32(str('0_01100000_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101001_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101010_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101011_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101100_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101101_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101110_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101111_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01110000_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01110001_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01110010_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01110011_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01110100_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01110101_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01110110_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01110111_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01111111_00000000000000000000000').replace('_', '')),
            ],
            dtype=torch.float32,
            device=device()
        )
        out = x.view(dtype=torch.int32)
        result = torch.zeros(
            out.size(),
            dtype=torch.uint8,
            device=out.device
        )
        x = 4
        base = 104
        mask = _s1exmy_base_exponent_and_mask(out, result, x, base)
        msg_list = [
            [i, val.item(), mask.item()]
            for i, (val, mask) in enumerate(zip(result >> (7 - x), mask))
        ]
        msg = '\n'.join(map(str, msg_list))
        logger.info(
            f"x:{x} base:{base}\n{msg}"
        )
        result.zero_()

        x = 3
        base = 112
        mask = _s1exmy_base_exponent_and_mask(out, result, x, base)
        msg_list = [
            [i, val.item(), mask.item()]
            for i, (val, mask) in enumerate(zip(result >> (7 - x), mask))
        ]
        msg = '\n'.join(map(str, msg_list))
        logger.info(
            f"x:{x} base:{base}\n{msg}"
        )
        result.zero_()

        x = 2
        base = 116
        mask = _s1exmy_base_exponent_and_mask(out, result, x, base)
        msg_list = [
            [i, val.item(), mask.item()]
            for i, (val, mask) in enumerate(zip(result >> (7 - x), mask))
        ]
        msg = '\n'.join(map(str, msg_list))
        logger.info(
            f"x:{x} base:{base}\n{msg}"
        )
        result.zero_()

        x = 1
        base = 118
        mask = _s1exmy_base_exponent_and_mask(out, result, x, base)
        msg_list = [
            [i, val.item(), mask.item()]
            for i, (val, mask) in enumerate(zip(result >> (7 - x), mask))
        ]
        msg = '\n'.join(map(str, msg_list))
        logger.info(
            f"x:{x} base:{base}\n{msg}"
        )
        result.zero_()

        x = 0
        base = 119
        mask = _s1exmy_base_exponent_and_mask(out, result, x, base)
        msg_list = [
            [i, val.item(), mask.item()]
            for i, (val, mask) in enumerate(zip(result >> (7 - x), mask))
        ]
        msg = '\n'.join(map(str, msg_list))
        logger.info(
            f"x:{x} base:{base}\n{msg}"
        )
        result.zero_()

        #############################################################################################
        logger.info(
            f"开始测试_s1exmy_base_mantissa_and_bernoulli()..."
        )
        x = torch.tensor(
            [
                binstr_to_float32(str('0_00000000_00000000001000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00100000010000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01000000100000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01100001000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10000010000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10100100000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11001000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11110000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00000000001100000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00100000011000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01000000110000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01100001100000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10000011000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10100110000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11001100000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11111000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00000000001110000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00100000011100000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01000000111000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01100001110000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10000011100000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10100111000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11001110000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11111100000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00000000001111000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00100000011110000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01000000111100000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01100001111000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10000011110000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10100111100000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11001111000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11111110000000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00000000001111100000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_00100000011111000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01000000111110000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_01100001111100000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10000011111000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_10100111110000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11001111100000000000000').replace('_', '')),
                binstr_to_float32(str('0_00000000_11111111000000000000000').replace('_', '')),
            ],
            dtype=torch.float32,
            device=device()
        )
        out = x.view(dtype=torch.int32)
        result = torch.zeros(
            out.size(),
            dtype=torch.uint8,
            device=out.device
        )
        y = 3
        mask = _s1exmy_base_mantissa_and_bernoulli(out, result, y)
        msg_list = [
            [i, val.item(), mask.item()]
            for i, (val, mask) in enumerate(zip(result & ((1 << y) - 1), mask))
        ]
        msg = '\n'.join(map(str, msg_list))
        logger.info(
            f"y:{y}\n{msg}"
        )
        result.zero_()

        y = 2
        mask = _s1exmy_base_mantissa_and_bernoulli(out, result, y)
        msg_list = [
            [i, val.item(), mask.item()]
            for i, (val, mask) in enumerate(zip(result & ((1 << y) - 1), mask))
        ]
        msg = '\n'.join(map(str, msg_list))
        logger.info(
            f"y:{y}\n{msg}"
        )
        result.zero_()

        y = 7
        mask = _s1exmy_base_mantissa_and_bernoulli(out, result, y)
        msg_list = [
            [i, val.item(), mask.item()]
            for i, (val, mask) in enumerate(zip(result & ((1 << y) - 1), mask))
        ]
        msg = '\n'.join(map(str, msg_list))
        logger.info(
            f"y:{y}\n{msg}"
        )
        result.zero_()

        #############################################################################################
        t = torch.tensor(
            [
                binstr_to_float32(str('0_00000000_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('1_00000000_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01111111_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('1_01111111_00000000000000000000000').replace('_', '')),

                # sign == 0 exponent == 104
                binstr_to_float32(str('0_01101000_00011111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_00111111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_01011111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_01111111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_10011111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_10111111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_11011111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_11111111111111100000000').replace('_', '')),

                # sign == 0 exponent == 104
                binstr_to_float32(str('0_01101000_00000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_00100000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_01000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_01100000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_10000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_10100000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_11000000000000000000000').replace('_', '')),
                binstr_to_float32(str('0_01101000_11100000000000000000000').replace('_', '')),

                # sign == 0 exponent == 119
                binstr_to_float32(str('0_01110111_00011111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01110111_00111111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01110111_01011111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01110111_01111111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01110111_10011111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01110111_10111111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01110111_11011111111111100000000').replace('_', '')),
                binstr_to_float32(str('0_01110111_11111111111111100000000').replace('_', '')),

                # sign == 1 exponent == 104
                binstr_to_float32(str('1_01101000_00011111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01101000_00111111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01101000_01011111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01101000_01111111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01101000_10011111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01101000_10111111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01101000_11011111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01101000_11111111111111100000000').replace('_', '')),

                # sign == 1 exponent == 119
                binstr_to_float32(str('1_01110111_00011111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01110111_00111111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01110111_01011111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01110111_01111111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01110111_10011111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01110111_10111111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01110111_11011111111111100000000').replace('_', '')),
                binstr_to_float32(str('1_01110111_11111111111111100000000').replace('_', '')),
            ],
            dtype=torch.float32,
            device=device()
        )
        x = 4
        y = 3
        base = 104
        result = _s1exmy_base(t, x, y, base)
        logger.info(
            f"开始测试_s1exmy_base()...\n"
            f"x:{x} y:{x} base:{base}\n{result}"
        )
        #############################################################################################
        t = torch.tensor(
            [i for i in range(256)],
            dtype=torch.uint8,
            device=device()
        )
        x = 4
        y = 3
        base = 104
        result = torch.zeros(
            t.size(),
            dtype=torch.int32,
            device=t.device
        )
        _s1exmy_base_decode_sign(t, result)
        _s1exmy_base_decode_exponent(t, result, x, base)
        _s1exmy_base_decode_mantissa(t, result, y)
        mask = _s1exmy_base_decode_mask(t)
        result.masked_fill_(mask, 0)
        logger.info(
            f"开始测试_s1exmy_base_decode()...\n"
            f"x:{x} y:{x} base:{base}\n{result}\n{result.view(dtype=torch.float32)}"
        )
        if is_master():
            with open("./s1e4m3_104_map", 'w', encoding='utf-8') as f:
                for i, (num, mask) in enumerate(zip(result.view(dtype=torch.float32), mask)):
                    bits = f'{struct.unpack("!I", struct.pack("!f", num.item()))[0]:032b}'
                    bits = f"{bits[0:1]} {bits[1:9]} {bits[9:]}"
                    f.write(str(i) + ': ' + bits + f" {str(mask.item())}")
                    f.write('\n')

        #############################################################################################
        t = torch.zeros(
            2* 256 * 1024 * 1024,
            dtype=torch.float32,
            device=device()
        )
        x = 4
        y = 3
        base = 104
        memory_allocated_pre = torch.cuda.memory_allocated(device())
        memory_reserved_pre = torch.cuda.memory_reserved(device())
        result = _s1exmy_base(t, x, y, base)
        memory_allocated_post = torch.cuda.memory_allocated(device())
        memory_reserved_post = torch.cuda.memory_reserved(device())
        logger.info(
            f"开始测试_s1exmy_base()显存峰值占用...\n"
            f"x:{x} y:{y} base:{base}\n"
            f"pre-allocated: {memory_allocated_pre / (1024 ** 3):.2f}\n"
            f"pre-reserved: {memory_reserved_pre / (1024 ** 3):.2f}\n"
            f"post-allocated: {memory_allocated_post / (1024 ** 3):.2f}\n"
            f"post-reserved: {memory_reserved_post / (1024 ** 3):.2f}"
        )
        
        del t
        memory_allocated_pre = torch.cuda.memory_allocated(device())
        memory_reserved_pre = torch.cuda.memory_reserved(device())
        result = _s1exmy_base_decode_and_sum(
            [result, result],
            [torch.tensor([1.0], dtype=torch.float32, device=result.device),
             torch.tensor([1.0], dtype=torch.float32, device=result.device)],
            x,
            y,
            base
        )
        memory_allocated_post = torch.cuda.memory_allocated(device())
        memory_reserved_post = torch.cuda.memory_reserved(device())
        logger.info(
            f"开始测试_s1exmy_base_decode_and_sum()显存峰值占用...\n"
            f"x:{x} y:{y} base:{base}\n"
            f"pre-allocated: {memory_allocated_pre / (1024 ** 3):.2f}\n"
            f"pre-reserved: {memory_reserved_pre / (1024 ** 3):.2f}\n"
            f"post-allocated: {memory_allocated_post / (1024 ** 3):.2f}\n"
            f"post-reserved: {memory_reserved_post / (1024 ** 3):.2f}"
        )