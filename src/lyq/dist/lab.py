from enum import Enum, auto
from typing import (
    TypedDict,
    Callable,
)
import json
import hashlib
from functools import cached_property
import time
from pathlib import Path

from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.distributed import (
    ReduceOp,
)
from torch.nn.parallel.distributed import (
    DistributedDataParallel as DDP
)
from torch.optim import (
    AdamW,
)

from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast
)
from datasets import load_dataset

from lyq import (
    Configs,
    Logger
)
from lyq.dist import *
from lyq.dist.comm_hooks import *
from lyq.utils.file import *
from lyq.utils.data import (
    LyqDataLoaderIterator,
    LyqDataset,
    BatchTokendSample
)
from lyq.utils.trace import (
    TraceStepSchema
)
from lyq.optim import *

__all__ = [
    'LyqLab',
]

class LyqLab():
    """
    实验类

    total_steps: 训练过程的总步数  
    max_seq_len: 单个句子中token最大数量  
    micro_batch_size_per_device: int  
    micro_num_per_batch: int   
    optimizer: 优化器类型  
    lr_schduler: 学习率调度器类型  
    quantization: 梯度量化类型
    """

    class Model(Enum):
        QWEN2_5_0_5_B = auto()
    
    _model_id_map: dict[Model, str] = {
        Model.QWEN2_5_0_5_B: "Qwen/Qwen2.5-0.5B",
    }

    def _check_model(
        self,
        model: Model
    ) -> None:
        assert isinstance(model, LyqLab.Model), "模型参数传递不合法！"
        assert self._model_id_map.get(model), "实验的模型不存在！"

        self._model_id: str = self._model_id_map[model]
        self._model_dir: str = self.configs.output_dir + self._model_id + '/'
        self._main_dir: str = self._model_dir + 'main/'

        self.logger.info(
            f"处理模型参数中...\n"
            f"模型ID：{self._model_id}\n"
            f"模型目录路径：{self._model_dir}\n"
            f"main模型目录路径：{self._main_dir}"
        )
    
    class Optim(Enum):
        ADAMW = auto()
    
    class OptimSchema(TypedDict):
        lr: float
        betas: tuple[float, float]
        eps: float
        weight_decay: float
    
    def _check_optim(
        self,
        optim: Optim,
        optim_configs: OptimSchema
    ) -> None:
        assert isinstance(optim, LyqLab.Optim), "优化器参数传递不正确！"

        self._optim_type: LyqLab.Optim = optim
        self._optim_configs: LyqLab.OptimSchema = optim_configs

        self.logger.info(
            f"处理优化器参数中...\n"
            f"优化器类型：{self._optim_type.name}\n"
            f"优化器配置：{self._optim_configs}"
        )
    
    class LR(Enum):
        LWLDLR = auto() # Linear Warmup with Liner Decay
        LWCDLR = auto()
    
    class LRSchema(TypedDict):
        warmup_ratio: float
        min_lr_ratio: float
    
    def _check_lr(
        self,
        lr: LR,
        lr_configs: LRSchema
    ) -> None:
        assert isinstance(lr, LyqLab.LR), "调度器参数传递不正确！"

        self._lr_type: LyqLab.LR = lr
        self._lr_configs: LyqLab.LRSchema = lr_configs

        self.logger.info(
            f"处理调度器参数中...\n"
            f"调度器类型：{self._lr_type.name}\n"
            f"调度器配置：{self._lr_configs}"
        )
    
    class Quan(Enum):
        NOQUAN = auto()
        S1E4M3_104_QUAN = auto()
    
    _quan_map: dict[
        Quan,
        Callable[
            [
                dist.ProcessGroup,
                dist.GradBucket # type: ignore
            ],
            torch.futures.Future[torch.Tensor]
        ]
    ] = {
        Quan.NOQUAN: noquan_hook,
        Quan.S1E4M3_104_QUAN: s1e4m3_104_quan_hook,
    }

    def _check_quan(
        self,
        quan: Quan
    ) -> None:
        assert isinstance(quan, LyqLab.Quan), "量化参数传递不正确！"
        assert self._quan_map.get(quan), "实验的量化钩子不存在！"

        self._quan: LyqLab.Quan = quan
        if self._quan == LyqLab.Quan.NOQUAN:
            self._commmetrix = noquan_get_commmetrix()
        elif self._quan == LyqLab.Quan.S1E4M3_104_QUAN:
            self._commmetrix = s1e4m3_104_quan_get_commmetrix()
        self._comm_hook: Callable[
            [
                dist.ProcessGroup,
                dist.GradBucket # type: ignore
            ],
            torch.futures.Future[torch.Tensor]
        ] = self._quan_map[quan]

        self.logger.info(
            f"处理量化参数中...\n"
            f"量化类型：{self._quan.name}\n"
            f"量化钩子：{self._comm_hook.__name__}"
        )
    
    def _check_training_args(
        self,
        max_seq_len,
        total_steps,
        micro_num_per_batch,
        micro_batch_size_per_device
    ) -> None:
        self._max_seq_len = max_seq_len
        self._total_steps = total_steps
        self._world_size = world_size()
        self._micro_num_per_batch = micro_num_per_batch
        self._micro_batch_size_per_device = micro_batch_size_per_device

        self.logger.info(
            f"处理训练参数中...\n"
            f"训练集单个样本最大长度：{self._max_seq_len}\n"
            f"训练总步数：{self._total_steps}\n"
            f"数据并行规模：{self._world_size}卡\n"
            f"每步微批次数量：{self._micro_num_per_batch}\n"
            f"每微批次样本数量：{self._micro_batch_size_per_device}"
        )
    
    def _check_lab(
        self
    ) -> None:
        """
        需要经历前面的_check*之后才可以调用！！！
        """
        self._lab_dir: str = self._model_dir + self.id + '/'
        self._lab_desc_file: str = self._lab_dir + 'lab.json'
        self._lab_trace_file: str = self._lab_dir + 'trace.jsonl'

        self.logger.info(
            f"处理实验文件夹中...\n"
            f"实验文件夹路径：{self._lab_dir}\n"
            f"实验说明文件路径：{self._lab_desc_file}\n"
            f"实验轨迹文件路径：{self._lab_trace_file}"
        )
    
    def __init__(
        self,
        configs: Configs,
        logger: Logger,
        *,
        model: Model = Model.QWEN2_5_0_5_B,
        optim: Optim = Optim.ADAMW,
        optim_configs: OptimSchema = {
            "lr": 1e-4,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.01
        },
        lr: LR = LR.LWCDLR,
        lr_configs: LRSchema = {
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.05
        },
        quan: Quan = Quan.NOQUAN,
        max_seq_len: int = 1024,
        total_steps: int = 50000,
        micro_num_per_batch: int = 32,
        micro_batch_size_per_device: int = 1
    ) -> None:
        self.configs = configs
        self.logger = logger

        self._check_model(model)
        self._check_optim(
            optim,
            optim_configs
        )
        self._check_lr(
            lr,
            lr_configs
        )
        self._check_quan(quan)
        self._check_training_args(
            max_seq_len,
            total_steps,
            micro_num_per_batch,
            micro_batch_size_per_device
        )
        self._check_lab()
        _register_lab(self)

        self.logger.info(
            f"实验对象处理完成！\n"
            f"实验ID：{self.id}"
        )
    
    _main_files: list[str] = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json"
    ]
    
    @master_function
    def _check_main_model(self) -> None:
        """
        检查main模型
        """
        self.logger.info(f"检查main模型{self._model_id}文件中...")
        if not is_files_all_exists(
            self._main_dir,
            self._main_files
        ):
            self.logger.info(
                f"main模型{self._model_id}文件不完全！\n"
                 "从HF仓库中拉取..."
            )
            s = snapshot_download(
                self._model_id,
                allow_patterns=[
                    "*.json"
                ]
            )
            self.logger.info(
                f"从HF仓库中拉取模型{self._model_id}完成！\n"
                 "创建初始原始模型并保存中..."
            )
            AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(
                    s,
                    trust_remote_code=True
                ),
                trust_remote_code=True,
                dtype=torch.float32
            ).save_pretrained(
                self._main_dir,
                save_peft_format=False,
                save_original_format=False
            )

            AutoTokenizer.from_pretrained(
                s,
                trust_remote_code=True,
            ).save_pretrained(
                self._main_dir,
                save_peft_format=False,
                save_original_format=False
            )

            self.logger.info(
                f"原始模型{self._model_id}文件保存完成！\n"
                f"保存路径：{self._main_files}"
            )

        self.logger.info(f"检查main模型{self._model_id}文件完成！")
    
    @master_function
    def _check_lab_dir(self) -> None:
        self.logger.info(f"检查实验{self.id}文件夹是否存在中...")
        # TODO lab.json文件不存在的情况
        if not Path(self._lab_dir).exists():
            self.logger.info(
                f"实验{self.id}文件夹不存在！\n"
                f"创建实验文件夹中..."
            )
            Path(self._lab_dir).mkdir()
            with Path(self._lab_desc_file).open("w", encoding='utf-8') as f:
                f.write(self.__repr__())
            Path(self._lab_trace_file).touch()
        self.logger.info(f"检查实验{self.id}文件夹完成！")
    
    _checkpoint_files: list[str] = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "lr.pth",
        "optim.pth",
    ]
    
    def _init_where(self):
        self.logger.info("检查是否存在检查点文件中...")
        p = find_max_checkpoint(self._lab_dir)
        self._step: int = 0 # 记录模型执行的步数，0代表执行了0步，表明为初始状态
        if p is None:
            self.logger.info("没有发现检查点文件，使用main初始模型构建！")
            self._where = self._main_dir
            self._step = 0
        else:
            self.logger.info(f"发现检查点{p.name}，检查有效性中...")
            is_valid = True
            if not is_files_all_exists(str(p)+'/', self._checkpoint_files):
                is_valid = False
            
            if is_valid:
                self.logger.info(f"检查点{p.name}有效，使用此检查点构建！")
                self._where = str(p)
                self._step = int(p.name[-5:])
            else:
                self.logger.info(f"检查点{p.name}无效，使用main初始模型构建！")
                self._where = self._main_dir
                self._step = 0
        
        self.logger.info(f"检查{self._lab_trace_file}中的步数是否合法...")
        with Path(self._lab_trace_file).open("r", encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        assert line_count == self._step, f"trace文件中步数{line_count}和希望的步数{self._step}不相符"
        if is_master():
            self._trace_writer = Path(self._lab_trace_file).open("a", encoding='utf-8')
        self.logger.info(
            f"检查检查点文件完成！\n"
            f"实验加载目录路径：{self._where}\n"
            f"当前实验已走步数：{self._step}\n"
            f"记载训练踪迹文件路径：{self._lab_trace_file}"
        )
    
    def _load_model(self):
        pretrain: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self._where,
            trust_remote_code=True
        )
        pretrain.to(self._device) # type: ignore
        # pretrain.gradient_checkpointing_enable()
        self._model: DDP = DDP(
            pretrain,
            init_sync=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=20000,
        )
        self._model.register_comm_hook(None, self._comm_hook)
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self._main_dir,
            trust_remote_code=True
        )
    
    def _load_optim(self):
        if self._optim_type == LyqLab.Optim.ADAMW:
            self._optim = AdamW(
                self._model.named_parameters(),
                lr=self._optim_configs['lr'],
                betas=self._optim_configs["betas"],
                eps=self._optim_configs['eps'],
                weight_decay=self._optim_configs['weight_decay']
            )
        else:
            assert False, "_load_optim 不能进入这里！！！"
        if self._where != self._main_dir:
            self._optim.load_state_dict(
                torch.load(
                    self._optim_path,
                    map_location='cpu'
                )
            )
    
    def _load_lr(self):
        if self._lr_type == LyqLab.LR.LWLDLR:
            self._lr = LWLDLR(
                self._optim,
                warmup_iters=self._warmup_steps,
                total_iters=self._total_steps,
                last_epoch=self._step-1
            )
        elif self._lr_type == LyqLab.LR.LWCDLR:
            self._lr = LWCDLR(
                self._optim,
                warmup_iters=self._warmup_steps,
                total_iters=self._total_steps,
                min_lr_ratio=self._min_lr_ratio,
                last_epoch=self._step-1
            )
        else:
            assert False, "_load_lr 不能进入这里！！！"
        if self._where != self._main_dir:
            self._lr.load_state_dict(
                torch.load(
                    self._lr_path,
                    map_location='cpu'
                )
            )
    
    def _load_from_where(self):
        self._device: torch.device = device()
        self._optim_path = Path(self._where) / "optim.pth"
        self._lr_path = Path(self._where) / "lr.pth"
        self._warmup_steps = int(self._total_steps * self._lr_configs['warmup_ratio'])
        self._min_lr_ratio = self._lr_configs['min_lr_ratio']
        self.logger.info(
            f"基于{self._where}构建实验中...\n"
            f"从{self._where}加载模型：{self._model_id}\n"
            f"使用设备：{self._device}\n"
            f"向模型中加入量化：{self._quan.name}\n"
            f"加载优化器：{self._optim_type.name}\n"
            f"优化器配置：{self._optim_configs}\n"
            f"优化器加载路径：{self._optim_path}\n"
            f"加载学习率调度器：{self._lr_type.name}\n"
            f"学习率调度器配置：{self._lr_configs}\n"
            f"学习率调度器加载路径：{self._lr_path}\n"
            f"总迭代次数：{self._total_steps}\n"
            f"预热步数：{self._warmup_steps}\n"
            f"学习率最小值：{self._min_lr_ratio}"
        )
        self._load_model()
        self._load_optim()
        self._load_lr()
    
    def _prepare_data(self):
        self.logger.info("准备数据集和数据分配器中...")
        self._start = LyqDataLoaderIterator.step_to_start(
            self._step,
            self._micro_batch_size_per_device * self._micro_num_per_batch,
            self._world_size
        )
        self.logger.info(f"训练集跳过{self._start}条数据！")
        self._data_collator = DataCollatorForLanguageModeling(
            self._tokenizer,
            mlm=False
        )
        self._train_dataloader_iter = LyqDataLoaderIterator(
            self.configs.train_file,
            self._start,
            self._world_size,
            rank(),
            self._micro_batch_size_per_device * self._micro_num_per_batch,
            self._data_collator,
            LyqDataset.process_sample,
            self._tokenizer
        )

        self.logger.info("训练数据加载器和验证数据加载器构建完成！")
    
    def _prepare_for_train(self) -> None:
        """
        训练前准备工作
        """
        self.logger.info(
            "训练前准备工作进行中..."
        )
        self.logger.info(
            f"主进程开始检查模型以及实验文件！\n"
            f"其他进程barrier等待！"
        )
        with sync_scope():
            self._check_main_model()
            self._check_lab_dir()
        
        self.logger.info(
            f"主进程检查完成！开始准备加载实验！"
        )
        self._init_where()
        self._load_from_where()
        self._prepare_data()
    
    def _pre_step(self) -> None:
        self._optim.zero_grad()
        self.batch_data = { # type: ignore
            k: v.to(self._device) # type: ignore
            for k, v in next(self._train_dataloader_iter).items()
        } 

        # 指标
        self.train_loss = torch.tensor(
            [0.0], 
            dtype=torch.float, 
            device=self._device
        )
        self.tokens = torch.sum(
            self.batch_data['token_count'],
            dtype=torch.int32
        )
        self.lr = self._optim.param_groups[0]['lr']
    
    @master_function
    def _process_grad(self) -> None:
        # 收集梯度
        grads = []
        names = []
        for name, param in self._model.named_parameters():
            names.append(name)
            grads.append(param.grad.flatten()) # type: ignore
        all_grads = torch.cat(grads)
        self.grad_norm = torch.linalg.vector_norm(all_grads)

        if (self._step + 1) % 10 == 0:
            self.layer_dict = {}
            for name, grad in zip(names, grads):
                normalized_grad: torch.Tensor = grad / self.grad_norm
                out = normalized_grad.view(torch.int32)
                exponent = (out >> 23) & 0xFF
                bucket_counts = torch.bincount(exponent, minlength=127).cpu().tolist()
                self.layer_dict[name] = bucket_counts
        del all_grads
        del grads
    
    def _post_step(self) -> None:
        self._process_grad()

        self._optim.step()
        self._lr.step()
        self.train_time_end = time.time()

        self.train_time = self.train_time_end - self.train_time_begin
        dist.all_reduce(
            self.tokens
        )
        dist.all_reduce(
            self.train_loss,
            ReduceOp.AVG
        )
        self._commmetrix.all_reduce()

        self._push_trace()

        self.logger.info(
            f"step: {self._step}\n"
            f"lr: {self.lr}\n"
            f"avg_loss: {self.train_loss.item()}"
        )
        self._step += 1
    
    @master_function
    def _push_trace(self) -> None:
        trace: TraceStepSchema = {
            'step': self._step,
            'lr': self.lr,
            'train_time': self.train_time,
            'grad_sync_total_time': self._commmetrix.total_time,
            'grad_sync_comp_time': self._commmetrix.comp_time,
            'grad_sync_comm_time': self._commmetrix.comm_time,
            'grad_sync_comm_bytes': self._commmetrix.comm_bytes,
            'tokens': self.tokens.item(), # type: ignore
            'train_loss': self.train_loss.item(),
            'grad_norm': self.grad_norm.item(),
            'layer_dict': self.layer_dict if (self._step + 1) % 10 == 0 else {}
        }
        self._trace_writer.write(
            json.dumps(
                trace,
                ensure_ascii=False
            ) + '\n'
        )
        
    def _per_step(self) -> None:
        self._pre_step()

        self.train_time_begin = time.time()
        for i in range(self._micro_num_per_batch):
            micro_batch_data = {
                    k: v[(i * self._micro_batch_size_per_device): ((i + 1) * self._micro_batch_size_per_device)] # type: ignore 
                    for k, v in self.batch_data.items()
                }

            if i == self._micro_num_per_batch - 1:
                output: CausalLMOutputWithPast = self._model(**micro_batch_data)
                loss: torch.Tensor = output.loss # type: ignore
                loss = loss / self._micro_num_per_batch
                loss.backward()
            else:
                with self._model.no_sync():
                    output: CausalLMOutputWithPast = self._model(**micro_batch_data)
                    loss: torch.Tensor = output.loss # type: ignore
                    loss = loss / self._micro_num_per_batch
                    loss.backward()
            
            self.train_loss += loss
        self._post_step()
    
    def train(
        self,
        *,
        checkpoint_nums: int = 10,
        checkpoint_steps: int = 500,
    ) -> None:
        """
        继续训练checkpoint_nums个检查点
        """
        # assert checkpoint_nums >= 1, f"训练参数checkpoint_nums={checkpoint_nums}不合法！"
        # assert checkpoint_steps >= 10, f"训练参数checkpoint_steps={checkpoint_steps}不合法！"

        self._prepare_for_train()

        self.logger.info(
            f"开始训练...\n"
            f"当前步数：{self._step}"
        )
        for _ in range(checkpoint_nums):
            for _ in range(checkpoint_steps):
                self._per_step()
            self._save_checkpoint()
    
    def _save_model(self):
        self.logger.info("保存模型中...")
        self._model.module.save_pretrained(
            self._checkpoint_dir,
            save_peft_format=False,
            save_original_format=False
        )
        self.logger.info("保存模型完成！")
    
    def _save_optim(self):
        self.logger.info(f"保存优化器中...")
        torch.save(
            self._optim.state_dict(),
            self._checkpoint_dir + 'optim.pth'
        )
        self.logger.info(f"保存优化器完成！")
    
    def _save_lr(self):
        self.logger.info(f"保存学习率调度器中...")
        torch.save(
            self._lr.state_dict(),
            self._checkpoint_dir + 'lr.pth'
        )
        self.logger.info(f"保存学习率调度器完成！")
    
    def _save_checkpoint(self) -> None:
        with sync_scope():
            self._checkpoint_dir = self._lab_dir + 'checkpoint-' + f'{self._step:05d}/'
            self.logger.info(
                f"保存检查点目录路径：{self._checkpoint_dir}..."
            )
            if is_master():
                self._save_model()
                self._save_optim()
                self._save_lr()
    
    def verify(self) -> bool:
        """
        通过检查训练集中的词数量来检查训练是否有效
        """
        self.logger.info(
            f"准备检查实验{self.id}有效性..."
            f"准备分词器、数据集和数据分配器中..."
        )
        self._start = LyqDataLoaderIterator.step_to_start(
            0,
            self._micro_batch_size_per_device * self._micro_num_per_batch,
            self._world_size
        )
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self._main_dir,
            trust_remote_code=True
        )
        self._data_collator = DataCollatorForLanguageModeling(
            self._tokenizer,
            mlm=False
        )
        self._train_dataloader_iter = LyqDataLoaderIterator(
            self.configs.train_file,
            self._start,
            self._world_size,
            rank(),
            self._micro_batch_size_per_device * self._micro_num_per_batch,
            self._data_collator,
            LyqDataset.process_sample,
            self._tokenizer
        )
        self.logger.info("分词器、数据集和数据分配器构建完成！")

        self._trace_reader = Path(self._lab_trace_file).open("r", encoding='utf-8')

        self.logger.info(
            f"开始测试...\n"
            f"如需看详细进度请看对应的日志文件{self.logger.log_file}"
        )
        is_valid = True
        self._device: torch.device = device()
        for i, line in enumerate(self._trace_reader):
            trace_step: TraceStepSchema = json.loads(line)
            tokens = trace_step['tokens']
            batch_data: BatchTokendSample = next(self._train_dataloader_iter)
            real_tokens = torch.sum(batch_data['token_count']).to(self._device)
            dist.all_reduce(
                real_tokens
            )
            self.logger.debug(
                f"step: {i}: \n"
                f"train_tokens = {tokens}\n"
                f"real_tokens = {real_tokens.item()}"
            )
            if tokens != real_tokens.item():
                self.logger.error(
                    f"step: {i}: \n"
                    f"train_tokens = {tokens}\n"
                    f"real_tokens = {real_tokens.item()}"
                )
                is_valid = False
        return is_valid

    def __repr__(self) -> str:
        return json.dumps(
            {
                "model": self._model_id,
                "optim": self._optim_type.name,
                "optim_configs": self._optim_configs,
                "lr": self._lr_type.name,
                "lr_configs": self._lr_configs,
                "quan": self._quan.name,
                "max_seq_len": self._max_seq_len,
                "total_steps": self._total_steps,
                "world_size": self._world_size,
                "micro_num_per_batch": self._micro_num_per_batch,
                "micro_batch_size_per_device": self._micro_batch_size_per_device
            },
            ensure_ascii=False,
            indent=4
        )
    
    @cached_property
    def id(self) -> str:
        return hashlib.md5(self.__repr__().encode('utf-8')).hexdigest()

_current_lab: list[LyqLab] = []

def _register_lab(lab: LyqLab) -> None:
    global _current_lab
    assert not _current_lab, "目前只支持注册一个实验！"
    _current_lab.append(lab)