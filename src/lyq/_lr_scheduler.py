from typing_extensions import override
import math

import torch
from torch import (
    Tensor,
)
from torch.optim import (
    Optimizer,
    AdamW
)
from torch.optim.lr_scheduler import (
    LRScheduler,
    _warn_get_lr_called_within_step,
    _param_groups_val_list,
)

__all__ = [
    "LWLDLR",
    "LWCDLR"
]

class LWLDLR(LRScheduler):
    """
    实现线性预热+线性衰减学习率调度器类
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int = 5,
        total_iters: int = 100,
        last_epoch: int = -1,
    ) -> None:
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative.")
        
        if total_iters <= warmup_iters:
            raise ValueError("total_iters must be greater than warmup_iters to allow for a decay phase.")
        

        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.base_lrs: list[float | Tensor] = [
            group["lr"] for group in optimizer.param_groups
        ]
        super().__init__(optimizer, last_epoch)
    
    @override
    def get_lr(self) -> list[float | Tensor]:
        _warn_get_lr_called_within_step(self)

        if self.last_epoch < self.warmup_iters:
            return [
                base_lr * ((self.last_epoch + 1) / self.warmup_iters)
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch < self.total_iters:
            return [
                base_lr * ((self.total_iters - self.last_epoch - 1) / (self.total_iters - self.warmup_iters))
                for base_lr in self.base_lrs
            ]
        else:
            return _param_groups_val_list(self.optimizer, "lr")

class LWCDLR(LRScheduler):
    """
    实现线性预热+余弦衰减学习率调度器类
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int = 5,
        total_iters: int = 100,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative.")
        
        if total_iters <= warmup_iters:
            raise ValueError("total_iters must be greater than warmup_iters to allow for a decay phase.")
        

        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs: list[float | Tensor] = [
            group["lr"] for group in optimizer.param_groups
        ]
        super().__init__(optimizer, last_epoch)
    
    @override
    def get_lr(self) -> list[float | Tensor]:
        _warn_get_lr_called_within_step(self)

        if self.last_epoch < self.warmup_iters:
            return [
                base_lr * ((self.last_epoch + 1) / self.warmup_iters)
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch < self.total_iters:
            progress = (self.last_epoch + 1 - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            return [
                base_lr * self.min_lr_ratio + (base_lr - (base_lr * self.min_lr_ratio)) * (0.5 * (1.0 + math.cos(math.pi * progress)))
                for base_lr in self.base_lrs
            ]
        else:
            return _param_groups_val_list(self.optimizer, "lr")

if __name__ == '__main__':
    params = [
        torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float, device="cuda:0", requires_grad=True)
        for _ in range(5)
    ]
    optim = AdamW(
        params,
        lr=10,
    )
    lr = LWCDLR(
        optim,
        10,
        20
    )

    for i in range(25):

        print(f"{i+1}: lr={optim.param_groups[0]['lr']}")

        for param in params:
            y = torch.sum(param)
        optim.step()
        lr.step()
    
    # torch.save(
    #     {
    #         "lr_state_dict": lr.state_dict(),
    #         'optim_state_dict': optim.state_dict()
    #     },
    #     "./checkpoint.pth"
    # )

    # checkpoint = torch.load(
    #     "./checkpoint.pth",
    #     map_location='cpu'
    # )
    # optim.load_state_dict(checkpoint['optim_state_dict'])
    # lr.load_state_dict(checkpoint["lr_state_dict"])

    # for i in range(10, 20):

    #     print(f"{i+1}: lr={optim.param_groups[0]['lr']}")

    #     for param in params:
    #         y = torch.sum(param)
    #     optim.step()
    #     lr.step()