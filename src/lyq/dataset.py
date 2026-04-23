from pathlib import Path
from typing import (
    TypedDict, 
    TextIO,
    TypeAlias
)
import json
import logging
import time
import random

from datasets import load_dataset
from transformers import (
    PreTrainedTokenizer
)

from .config import *

__all__ = [
    'LyqDataset',
]

class StateItem(TypedDict):
    num_tokens: int
    num_samples: int

class StateI(TypedDict):
    total_tokens: int
    total_samples: int
    train: StateItem
    valid: StateItem

StateType: TypeAlias = dict[str, StateI]

class SampleSchema(TypedDict):
    text: str

class TokendSampleType(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    token_count: int

dataset_ids = {
    "Fineweb-edu": "HuggingFaceFW/fineweb-edu"
}

class LyqDataset:
    """
    自定义数据集类
    """
    def __init__(self) -> None:
        # 读取数据集状态文件
        state_file_path = Path(state_file())

        if not state_file_path.exists():
            with state_file_path.open('w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
        
        with state_file_path.open(encoding='utf-8') as f:
            self._state: StateType = json.load(f)
    
    # @staticmethod
    # def _empty_dataset() -> StateI:
    #     """

    #     """

    #     return {
    #         'total_samples': 0,
    #         'total_tokens': 0,
    #         'train':{
    #             'num_samples': 0,
    #             'num_tokens': 0
    #         },
    #         'valid':{
    #             'num_samples': 0,
    #             'num_tokens': 0
    #         }
    #     }
    
    # @staticmethod
    # def _write_samples_to_file(
    #     data: list,
    #     f: TextIO
    # ) -> tuple[int, int]:
    #     """

    #     """
    #     num_samples = 0
    #     num_tokens = 0
    #     for x in data:
    #         line = json.dumps(
    #             x,
    #             ensure_ascii=False
    #         )
    #         f.write(line + '\n')
    #         num_samples += 1
    #         num_tokens += x['token_count']
        
    #     return num_samples, num_tokens
    
    # @staticmethod
    # def _update_statei(
    #     train_num_samples: int,
    #     train_num_tokens: int,
    #     valid_num_samples: int,
    #     valid_num_tokens: int,
    #     statei: StateI
    # ) -> None:
    #     statei['total_samples'] += \
    #         train_num_samples + valid_num_samples
    #     statei['total_tokens'] += \
    #         train_num_tokens + valid_num_tokens
    #     statei['train']['num_samples'] += \
    #         train_num_samples
    #     statei['train']['num_tokens'] += \
    #         train_num_tokens
    #     statei['valid']['num_samples'] += \
    #         valid_num_samples
    #     statei['valid']['num_tokens'] += \
    #         valid_num_tokens
    
    # def _push_state(self) -> None:
    #     """

    #     """
    #     state_file_path = Path(state_file())
    #     with state_file_path.open('w', encoding='utf-8') as f:
    #         json.dump(self._state, f, ensure_ascii=False, indent=4)
        
    # def _download_from_hf_hub(
    #     self,
    #     name: str,
    #     configs: Configs,
    #     logger: logging.Logger,
    #     /,
    #     batch_size: int = 10000,
    #     batch_num: int = 10,
    #     ratio: float = 0.001
    # ) -> None:
    #     """

    #     Args:
    #         name: 处理的HF数据集名称
    #         configs: 配置文件
    #         logger: 日志对象
    #         /,
    #         batch_size: 一次处理的样本数量
    #         batch_num: 本次处理的批次总数
    #         ratio: 从每一批量中选取ratio份作为验证集
    #             默认999:1

    #     Returns:
    #         None
    #     """
    #     if self._state.get(name) is None:
    #         logger.info(f"发现新数据集{name}！")
    #         self._state[name] = self._empty_dataset()

    #     dataset_state: StateI = self._state[name]
    #     logger.debug(f"dataset_state: {dataset_state}")

    #     logger.info(f"从 HF Hub 加载数据集{name}中...")
    #     dataset = load_dataset(
    #         path=name,
    #         split="train",
    #         streaming=True
    #     )
    #     logger.info("加载数据集成功！")

    #     train_file = Path(configs.train_file)
    #     valid_file = Path(configs.valid_file)
    #     with train_file.open('a', encoding='utf-8') as train, \
    #          valid_file.open('a', encoding='utf-8') as valid:
            
    #         logger.info("跳过已加载数据")
    #         dataset.skip(dataset_state['total_samples'])

    #         logger.info("读取数据集中...")
    #         batch_samples: list = []
    #         cnt = 0

    #         end = batch_size - int(batch_size * ratio)
    #         logger.debug(f"训练集写入范围[:{end}]\n"
    #                      f"验证集集写入范围[:{end}]")

    #         # total_start: 记录全部批次处理总时间
    #         # start: 记录单一批次处理时间
    #         total_start = start = time.time()

    #         for sample in dataset:
    #             # 加载样本
    #             batch_samples.append(sample)
    #             cnt += 1

    #             # 是否读取了一个批次的样本
    #             if cnt >= batch_size:
    #                 logger.info(f"读取{batch_size}个样本")

    #                 logger.info("写入训练数据中...")
    #                 train_num_samples, train_num_tokens = \
    #                     self._write_samples_to_file(
    #                         batch_samples[:end],
    #                         train
    #                     )
    #                 logger.info("写入训练数据完成！\n"
    #                             f"samples: {train_num_samples}\n"
    #                             f"tokens: {train_num_tokens}\n")

    #                 logger.info("写入验证数据中...")
    #                 valid_num_samples, valid_num_tokens = \
    #                     self._write_samples_to_file(
    #                         batch_samples[end:],
    #                         valid
    #                     )
    #                 logger.info("写入验证数据完成！\n"
    #                             f"samples: {valid_num_samples}\n"
    #                             f"tokens: {valid_num_tokens}\n")
                    
    #                 self._update_statei(
    #                     train_num_samples,
    #                     train_num_tokens,
    #                     valid_num_samples,
    #                     valid_num_tokens,
    #                     dataset_state
    #                 )

    #                 batch_num -= 1
    #                 time_end = time.time()
    #                 logger.info(f"耗时:{time_end - start:.4f}s\n"
    #                             f"剩余批量：{batch_num}")
                    
    #                 if batch_num == 0:
    #                     total_end = time.time()
    #                     break

    #                 batch_samples.clear()
    #                 cnt = 0
    #                 start = time.time()
            
    #         logger.info("读取数据集完成！"
    #                     f"总耗时:{total_end - total_start:.4f}s")
        
    #     logger.info("更新状态文件中...")
    #     self._push_state(configs)
    #     logger.info("更新状态文件成功！")

    # def download_from_hf_hub(
    #     self,
    #     name: str,
    #     configs: Configs,
    #     logger: logging.Logger,
    #     /,
    #     batch_size: int = 10000,
    #     batch_num: int = 10,
    #     ratio: float = 0.001
    # ) -> None:
    #     """

    #     """
    #     self._download_from_hf_hub(
    #         name,
    #         configs,
    #         logger,
    #         batch_size=batch_size,
    #         batch_num=batch_num,
    #         ratio=ratio
    #     )
    #     time.sleep(10) # 给一点时间用于释放
    
    # @staticmethod
    # def create_valid_subset(
    #     configs: Configs,
    #     logger: logging.Logger,
    #     *,
    #     num_samples: int = 1024,
    #     seed: int = 42
    # ) -> None:
    #     logger.info("创建验证子集中...")

    #     logger.info("读取验证集中...")

    #     valid_file = Path(configs.valid_file)
    #     with valid_file.open(encoding='utf-8') as f:
    #         lines = f.readlines()

    #     logger.info("读取验证集完成！")

    #     random.seed(seed)
    #     subset = random.sample(lines, num_samples)
    #     small_valid_file = Path(configs.small_valid_file)
    #     with small_valid_file.open('w', encoding='utf-8') as f:
    #         f.writelines(subset)
        
    #     logger.info("创建验证子集完成！")
    
    @staticmethod
    def process_sample(
        sample: SampleSchema,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 1024
    ):
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