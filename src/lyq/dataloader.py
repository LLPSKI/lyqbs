from typing import TypedDict

from torch import (
    Tensor
)
import torch.distributed as dist

from datasets import (
    IterableDataset
)
from transformers import (
    DataCollatorForLanguageModeling
)

__all__ = [
    'BatchSchema',
    'LyqDataLoaderIterator',
    'LyqDataLoader'
]

class BatchSchema(TypedDict):
    token_count: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor

class LyqDataLoaderIterator:
    def __init__(
        self,
        dataset: IterableDataset,
        data_collator: DataCollatorForLanguageModeling,
        batch_size_per_device: int,
        world_size: int,
        rank: int,
        *,
        total: int,
    ) -> None:
        self.iter = iter(dataset)
        self.data_collator = data_collator
        self.batch_size_per_device = batch_size_per_device
        self.world_size = world_size
        self.rank = rank
        self._total = total
        self._passed = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> BatchSchema:
        if self._passed == self._total:
            raise StopIteration
        batch_data = []
        for i in range(self.world_size):
            if i == self.rank:
                for j in range(self.batch_size_per_device):
                    batch_data.append(next(self.iter))
            else:
                for j in range(self.batch_size_per_device):
                    next(self.iter)
            self._passed += self.batch_size_per_device
        return self.data_collator(batch_data)

class LyqDataLoader:
    """

    """
    def __init__(
        self,
        dataset: IterableDataset,
        data_collator: DataCollatorForLanguageModeling,
        batch_size_per_device: int,
        world_size: int,
        rank: int,
        total: int = -1,
    ) -> None:
        """
        初始化LyqDataLoader对象
        """
        self.dataset = dataset
        self.data_collator = data_collator
        self.batch_size_per_device = batch_size_per_device
        self.world_size = world_size
        self.rank = rank
        self.total = total
    
    def __iter__(self):
        return LyqDataLoaderIterator(
            self.dataset,
            self.data_collator,
            self.batch_size_per_device,
            self.world_size,
            self.rank,
            total=self.total
        )
    
    @staticmethod
    def step_to_start(
        step: int,
        batch_size_per_device: int,
        world_size: int
    ) -> int:
        return step * (world_size * batch_size_per_device)