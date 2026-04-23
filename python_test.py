from typing import (
    TypedDict,
    Callable,
    Any
)
import json
from itertools import (
    islice,
)

from datasets import load_dataset
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
import torch
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

class BatchSchema(TypedDict):
    token_count: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor

class SampleSchema(TypedDict):
    text: str

class TokendSampleType(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    token_count: int

def process_sample(
    sample: SampleSchema,
    tokenizer: PreTrainedTokenizer,
    max_len: int = 1024
) -> TokendSampleType:
    """

    """
    tokend: TokendSampleType = tokenizer(
        sample['text'],
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_attention_mask=True
    ) # type: ignore
    
    tokend['token_count'] = tokend['attention_mask'].count(1)
    return tokend

class LyqDataLoaderIterator:
    """
    自定义的多进程数据迭代器
    """
    def __init__(
        self,
        dataset_path: str,
        start: int,
        world_size: int,
        rank: int,
        batch_size_per_device: int,
        data_collator: DataCollatorForLanguageModeling,
        process_sample: Callable[
            [SampleSchema, PreTrainedTokenizer, int], 
            TokendSampleType
        ],
        tokenizer: PreTrainedTokenizer,
        max_len: int = 1024,
        total: int = -1
    ) -> None:
        """
        Args:
            dataset_path: 数据集.jsonl文件路径   
            world_size: 进程总数量  
            rank: 当前进程编号  
            batch_size_per_device: 每个进程需要的数据批量大小
            data_collator: 用于批量数据处理  
            total: 如果为-1代表无限行数，反之到达后抛出异常
        """
        self.dataset_path = dataset_path
        self.start = start
        self.f = open(
            self.dataset_path,
            'r',
            encoding='utf-8'
        )
        self.dataset = islice(self.f, self.start, None)
        self.world_size = world_size
        self.rank = rank
        self.total = total
        self.batch_size_per_device = batch_size_per_device
        self.data_collator = data_collator
        self.process_sample = process_sample
        self.tokenzier = tokenizer
        self.max_len = max_len
        self.passed = 0
    
    def __next__(self) -> BatchSchema:
        if self.passed == self.total:
            raise StopIteration
        batch_data = []
        for i in range(self.world_size):
            if i == self.rank:
                for _ in range(self.batch_size_per_device):
                    tokend = self.process_sample(
                        json.loads(next(self.dataset)),
                        self.tokenzier,
                        self.max_len
                    )
                    batch_data.append(tokend)
            else:
                for _ in range(self.batch_size_per_device):
                    next(self.dataset)
            self.passed += self.batch_size_per_device
        return self.data_collator(batch_data)
    
    @staticmethod
    def step_to_start(
        step: int,
        batch_size_per_device: int,
        world_size: int
    ) -> int:
        return step * (world_size * batch_size_per_device)

# class LyqDataLoader:
#     """
#     自定义的多进程数据分配器
#     """
#     def __init__(
#         self,
#         dataset_path: str,
#         start: int,
#         data_collator: DataCollatorForLanguageModeling,
#         process_sample: Callable[[Any], TokendSampleType],
#         tokenizer: PreTrainedTokenizer,
#         batch_size_per_device: int,
#         world_size: int,
#         rank: int,
#         max_len: int = 1024,
#         total: int = -1,
#     ) -> None:
#         """
#         初始化LyqDataLoader对象  

#         Args:  
#             dataset_path: 数据集.jsonl文件路径
#             start: 如果不为0，则代表需要跳过开头的start行
#         """
#         self.dataset_path = dataset_path
#         self.start = start
#         self.data_collator = data_collator
#         self.process_sample = process_sample
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.batch_size_per_device = batch_size_per_device
#         self.world_size = world_size
#         self.rank = rank
#         self.total = total
    
#     def __iter__(self):
#         return LyqDataLoaderIterator(
#             self.dataset_path,
#             self.start,
#             self.world_size,
#             self.rank,
#             self.batch_size_per_device,
#             self.data_collator,
#             self.process_sample,
#             self.tokenizer,
#             self.total
#         )
    
#     @staticmethod
#     def step_to_start(
#         step: int,
#         batch_size_per_device: int,
#         world_size: int
#     ) -> int:
#         return step * (world_size * batch_size_per_device)



if __name__ == '__main__':
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "/mnt/hdd2/liuyuqi/output/Qwen/Qwen2.5-0.5B/main/",
        trust_remote_code=True
    )
    # my_dataset = load_dataset(
    #     path='json',
    #     data_files={
    #         "train": '/mnt/hdd2/liuyuqi/dataset/train.jsonl',
    #     },
    #     streaming=True
    # )
    step = 40
    start = LyqDataLoaderIterator.step_to_start(
        step,
        32,
        2
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )
    train_dataloader_rank0 = LyqDataLoaderIterator(
        '/mnt/hdd2/liuyuqi/dataset/train.jsonl',
        start,
        2,
        0,
        32,
        data_collator,
        process_sample,
        tokenizer,
    )
    train_dataloader_rank1 = LyqDataLoaderIterator(
        '/mnt/hdd2/liuyuqi/dataset/train.jsonl',
        start,
        2,
        1,
        32,
        data_collator,
        process_sample,
        tokenizer,
    )


    with open('./tokens4.jsonl', 'w', encoding='utf-8') as f:
        for i in range(step, 10000):
            print(i)

            # rank0
            batch_data0 = next(train_dataloader_rank0)
            rank0_tokens = int(torch.sum(
                batch_data0['token_count'],
                dtype=torch.int32
            ).item())

            # rank1
            batch_data1 = next(train_dataloader_rank1)
            rank1_tokens = int(torch.sum(
                batch_data1['token_count'],
                dtype=torch.int32
            ).item())

            f.write(
                json.dumps(
                    {
                        f'step: {i}': rank0_tokens + rank1_tokens
                    }
                ) + '\n'
            )
            f.write(repr(batch_data0) + '\n')
            f.write(repr(batch_data1) + '\n')

            if i > step + 10:
                break