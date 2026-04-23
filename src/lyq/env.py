import os
from functools import (
    cached_property,
    cache,
    wraps,
)
from contextlib import (
    contextmanager,
)
import json
from typing import (
    Callable,
    Any
)

import torch
import torch.distributed as dist
from torch.accelerator import (
    current_accelerator,
)

__all__ = [
    'sync_scope',
    'global_env',
    'world_size',
    'rank',
    'is_master',
    'device',
    'master_function'
]

class _Env:
    """
    分布式环境类
    """
    def __init__(
        self,
        master_rank: int = 0,
    ) -> None:
        assert not dist.is_initialized(), "在构造Env对象之前不可以初始化分布式环境！"
        assert os.environ.get('RANK') is not None, "没有注入环境变量RANK！\n请使用torchrun启动Python脚本！"
        assert os.environ.get('WORLD_SIZE') is not None, "没有注入环境变量WORLD_SIZE！\n请使用torchrun启动Python脚本！"
        
        self._rank = int(os.environ['RANK'])
        self._world_size = int(os.environ['WORLD_SIZE'])
        self._is_master = self._rank == master_rank
        self._enable_profiling = True if os.environ.get('ENABLE_PROFILING') is not None else False

        _device = current_accelerator()
        assert _device is not None, "当前环境或Pytorch不支持加速器！"

        self._device = torch.device(
            type=_device.type if isinstance(_device, torch.device) else '',
            index=self._rank
        )
    
    def __enter__(self):
        dist.init_process_group(device_id=self._device)

        assert self._rank == dist.get_rank(), "初始化分布式环境后rank不对等！"
        assert self._world_size == dist.get_world_size(), "初始化分布式环境后world_size不对等！"
        return self
    
    def __exit__(self, exc_type, exc, tb):
        dist.destroy_process_group()
    
    def __getstate__(self) -> dict:
        return {
            "world_size": self.world_size,
            "is_master": self.is_master,
            "device": self.device.type
        }
    
    def __repr__(self) -> str:
        return json.dumps(
            {
                "rank": self.rank,
                "world_size": self.world_size,
                "is_master": self.is_master,
                "device": str(self.device)
            },
            ensure_ascii=False
        )
    
    @cached_property
    def rank(self) -> int:
        return self._rank
    
    @cached_property
    def is_master(self) -> bool:
        return self._is_master
    
    @cached_property
    def world_size(self) -> int:
        return self._world_size
    
    @cached_property
    def enable_profiling(self) -> bool:
        return self._enable_profiling
    
    @cached_property
    def device(self) -> torch.device:
        return self._device

_env = _Env()

@cache
def global_env() -> _Env:
    return _env
@cache
def world_size() -> int:
    return _env.world_size
@cache 
def rank() -> int:
    return _env.rank
@cache
def is_master() -> bool:
    return _env.is_master
@cache
def enable_profiling() -> bool:
    return _env.enable_profiling
@cache
def device() -> torch.device:
    return _env.device

def master_function(
    func: Callable[[Any], None]
) -> Callable[[Any], None]:
    """
    装饰器：用于修饰只需要主进程执行的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> None:
        if is_master():
            func(*args, **kwargs)
        return None
    return wrapper

@contextmanager
def sync_scope():
    """
    多进程同步区域
    """
    try:
        yield
    finally:
        dist.barrier()