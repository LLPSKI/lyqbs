import struct

__all__ = [
    'binstr_to_float32',
]

def binstr_to_float32(
    binstr: str
) -> float:
    """
    将一串32位的字符串转化为32位浮点数类型并返回
    """
    assert len(binstr) == 32, "输入的二进制字符串长度必须是32位！"

    byte_data = bytes(
        [int(binstr[i:i+8], 2) for i in range(0, 32, 8)]
    )

    return struct.unpack(
        '!f',
        byte_data
    )[0]