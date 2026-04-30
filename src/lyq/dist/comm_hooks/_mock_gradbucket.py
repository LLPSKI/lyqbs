import torch

__all__ = [
    "MockGradBucket",
]

class MockGradBucket:
    """
    用于验证通信钩子正确性的自定义梯度桶类
    """
    def __init__(
        self,
        tensor: torch.Tensor
    ) -> None:
        # tmps = []
        # for tensor in tensors:
        #     tmps.append(tensor.flatten())
        self._buffer = tensor
    
    def buffer(self) -> torch.Tensor:
        return self._buffer