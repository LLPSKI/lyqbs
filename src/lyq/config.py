from pathlib import Path
import json
from typing import TypedDict
from functools import cached_property

__all__ = [
    'Configs'
]

class _ConfigSchema(TypedDict):
    r"""
    配置项

    - project_name: 项目名称  
    - base_dir: 存放运行日志文件  
        - log/
            - rank0.log
            - rank1.log
            - ...
    - data_dir: 存放初始模型和模型训练检查点  
        - output/
            - Qwen/
                - Qwenx-yB-z/
                    - main/
                        - config.json
                        - generation_config.json
                        - model.safetensors
                        - tokenizer_config.json
                        - tokenizer.json
                    - lab_id/
                        - checkpoint-00100/
                            - ...
                        - trace.jsonl  
                        - lab.json
                    - lab_id/
                        - ...
            - xxx/
                - ...  
        - dataset/
            - state.json  
            - train.jsonl
            - valid.jsonl
            - small_vaild.jsonl
    """
    project_name: str
    base_dir: str
    data_dir: str

class Configs:
    """
    配置类

    - project_name
    - base_dir
        - state_file
        - log_file
    - data_dir
        - output_dir
        - dataset_dir
            - train_file
            - valid_file
            - small_valid_file
    """

    _default_config: _ConfigSchema = {
        'project_name': "lyqbs",
        'base_dir': "./",
        'data_dir': "./"
    }

    def __init__(
        self,
        path: Path | str | None = None
    ) -> None:
        """
        读取配置文件，初始化配置类  

        Args:  
            path: 默认下位于当前脚本执行目录下
        """

        if path is None:
            path = Path('./config.json') # 默认配置文件在脚本执行目录下
        else:
            path = path if isinstance(path, Path) else Path(path)

        if not path.exists():
            with path.open("w", encoding='utf-8') as f:
                json.dump(
                    self._default_config,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

        with path.open(encoding='utf-8') as f:
            self._configs: _ConfigSchema = json.load(f)
    
    @cached_property
    def project_name(self) -> str:
        return self._configs['project_name']
    
    @cached_property
    def base_dir(self) -> str:
        return self._configs['base_dir']
    @cached_property
    def log_dir(self) -> str:
        return self._configs['base_dir'] + 'log/'
    
    @cached_property
    def data_dir(self) -> str:
        return self._configs['data_dir']
    @cached_property
    def output_dir(self) -> str:
        return self.data_dir + 'output/'
    @cached_property
    def dataset_dir(self) -> str:
        return self.data_dir + 'dataset/'
    @cached_property
    def state_file(self) -> str:
        return self.dataset_dir + 'state.json'
    @cached_property
    def train_file(self) -> str:
        return self.dataset_dir + 'train.jsonl'
    @cached_property
    def valid_file(self) -> str:
        return self.dataset_dir + 'valid.jsonl'
    @cached_property
    def small_valid_file(self) -> str:
        return self.dataset_dir + 'small_valid.jsonl'