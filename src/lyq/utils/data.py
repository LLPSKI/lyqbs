from pathlib import Path
import json
from typing import (
    TypedDict,
    TypeAlias,
    TextIO,
    Callable
)
from enum import (
    Enum,
    auto
)
import time
import random
from itertools import islice

import torch

from transformers import (
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import (
    load_dataset,
    IterableDataset
)

from lyq.config import Configs
from lyq.log import Logger

__all__ = [
    "LyqDataset",
    "LyqDataLoaderIterator"
]

class SampleSchema(TypedDict):
    text: str
    id: str
    dump: str
    url: str
    data: str
    file_path: str
    language: str
    language_score: float
    token_count: int
    score: float
    int_score: int

class TokendSampleType(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    token_count: int

class BatchTokendSample(TypedDict):
    token_count: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

class StateI(TypedDict):
    num_tokens: int
    num_samples: int

class StateItem(TypedDict):
    total_tokens: int
    total_samples: int
    train: StateI
    valid: StateI

StateType: TypeAlias = dict[str, StateItem]

class LyqDataset:
    """
    自定义数据集类
    """
    def __init__(
        self,
        configs: Configs,
        logger: Logger
    ) -> None:
        self.configs = configs
        self.logger = logger

        assert self.logger.is_multirank == False, "LyqDataset只支持一个进程状态！"

        # 读取数据集状态文件
        dataset_dir_path = Path(self.configs.dataset_dir)
        state_file_path = Path(self.configs.state_file)

        if not dataset_dir_path.exists():
            dataset_dir_path.mkdir(exist_ok=True)

        if not state_file_path.exists():
            with state_file_path.open('w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
        
        with state_file_path.open(encoding='utf-8') as f:
            self.state: StateType = json.load(f)
    
    class DatasetID(Enum):
        FINEWEB_EDU = auto()
    
    _dataset_id_map: dict[DatasetID, str] = {
        DatasetID.FINEWEB_EDU: "HuggingFaceFW/fineweb-edu"
    }

    @staticmethod
    def _empty_stateitem() -> StateItem:
        """
        返回新数据对象
        """
        return {
            'total_samples': 0,
            'total_tokens': 0,
            'train':{
                'num_samples': 0,
                'num_tokens': 0
            },
            'valid':{
                'num_samples': 0,
                'num_tokens': 0
            }
        }

    @staticmethod
    def _write_samples_to_file(
        data: list,
        f: TextIO
    ) -> tuple[int, int]:
        """
        将批量数据一起写入文件
        """
        num_samples = 0
        num_tokens = 0
        for x in data:
            line = json.dumps(
                x,
                ensure_ascii=False
            )
            f.write(line + '\n')
            num_samples += 1
            num_tokens += x['token_count']
        
        return num_samples, num_tokens

    @staticmethod
    def _update_stateitem(
        train_num_samples: int,
        train_num_tokens: int,
        valid_num_samples: int,
        valid_num_tokens: int,
        statei: StateItem
    ) -> None:
        statei['total_samples'] += train_num_samples + valid_num_samples
        statei['total_tokens'] += train_num_tokens + valid_num_tokens
        statei['train']['num_samples'] += train_num_samples
        statei['train']['num_tokens'] += train_num_tokens
        statei['valid']['num_samples'] += valid_num_samples
        statei['valid']['num_tokens'] += valid_num_tokens
    
    def _push_state(self) -> None:
        state_file_path = Path(self.configs.state_file)
        with state_file_path.open('w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=4)

    def download_from_hf_hub(
        self,
        dataset_id: DatasetID,
        /,
        batch_size: int = 10000,
        batch_num: int = 10,
        ratio: float = 0.001
    ) -> None:
        """
        从HF中下载远程数据集到本地

        Args:
            dataset_id: 处理的HF数据集ID  
            /,  
            batch_size: 一次处理的样本数量  
            batch_num: 本次处理的批次总数  
            ratio: 从每一批量中选取ratio份作为验证集  
                默认999:1
        
        Returns:  
            None
        """
        assert self._dataset_id_map.get(dataset_id), "HF数据集ID不合法！"

        self.dataset_name = self._dataset_id_map[dataset_id]
        if self.state.get(self.dataset_name) is None:
            self.logger.info(f"发现新数据集{self.dataset_name}！")
            self.state[self.dataset_name] = LyqDataset._empty_stateitem()
        
        stateitem: StateItem = self.state[self.dataset_name]
        
        self.logger.info(f"从 HF Hub 加载数据集{self.dataset_name}中...")
        dataset: IterableDataset = load_dataset(
            self.dataset_name,
            split='train',
            streaming=True
        )
        self.logger.info("加载数据集成功！")

        train_file_path = Path(self.configs.train_file)
        valid_file_path = Path(self.configs.valid_file)
        with train_file_path.open('a', encoding='utf-8') as train, \
             valid_file_path.open('a', encoding='utf-8') as valid:
            
            self.logger.info(f"跳过{stateitem['total_samples']}条已加载数据！")
            dataset = dataset.skip(stateitem['total_samples'])

            self.logger.info("读取数据集中...")

            batch_samples: list = []
            cnt = 0
            end = batch_size - int(batch_size * ratio)
            self.logger.info(
                f"训练集写入范围[:{end}]\n"
                f"验证集集写入范围[{end}:]"
            )

            # total_start: 记录全部批次处理总时间
            # start: 记录单一批次处理时间
            total_start = start = time.time()

            for sample in dataset:
                batch_samples.append(sample)
                cnt += 1

                if cnt >= batch_size:
                    self.logger.info(
                        f"读取{batch_size}个样本\n"
                        f"写入训练集和验证集中..."
                    )
                    train_num_samples, train_num_tokens = self._write_samples_to_file(
                        batch_samples[:end],
                        train
                    )
                    valid_num_samples, valid_num_tokens = self._write_samples_to_file(
                        batch_samples[end:],
                        valid
                    )
                    self.logger.info(
                        f"写入训练集和验证集完成！"
                    )

                    self._update_stateitem(
                        train_num_samples,
                        train_num_tokens,
                        valid_num_samples,
                        valid_num_tokens,
                        stateitem
                    )

                    batch_num -= 1
                    time_end = time.time()
                    self.logger.info(
                        f"耗时:{time_end - start:.4f}s\n"
                        f"剩余批量：{batch_num}"
                    )

                    if batch_num == 0:
                        total_end = time.time()
                        break

                    batch_samples.clear()
                    cnt = 0
                    start = time.time()
            
            self.logger.info(
                f"读取数据集完成！\n"
                f"总耗时:{total_end - total_start:.4f}s"
            )

        self._push_state()
        time.sleep(10) # 给一点时间用于释放
    
    def create_valid_subset(
        self,
        *,
        num_samples: int = 1024,
        seed: int = 42
    ) -> None:
        self.logger.info("创建验证子集中...")

        valid_file = Path(self.configs.valid_file)
        with valid_file.open(encoding='utf-8') as f:
            lines = f.readlines()

        random.seed(seed)
        subset = random.sample(lines, num_samples)
        small_valid_file = Path(self.configs.small_valid_file)
        with small_valid_file.open('w', encoding='utf-8') as f:
            f.writelines(subset)
        
        self.logger.info("创建验证子集完成！")
    
    @staticmethod
    def process_sample(
        sample: SampleSchema,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 1024
    ) -> TokendSampleType:
        """
        对每一个原始样本进行分词处理
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
    
    def __next__(self) -> BatchTokendSample:
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