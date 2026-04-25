import logging
from pathlib import Path

from lyq.config import Configs

__all__ = [
    'Logger'
]

class Logger:
    """
    项目日志类
    """
    def __init__(
        self,
        configs: Configs,
        *,
        is_multirank: bool = False,
        rank: int | None = None,
        is_master: bool | None = None
    ) -> None:
        """
        设置全局统一的日志对象  
        每个进程有一个独立的日志对象，不同的日志文件
        主进程具有打印到终端的能力，其他进程只能写入到日志文件中
        """
        self.configs = configs
        self.is_multirank = is_multirank
        
        if self.is_multirank:
            assert rank is not None, "当使用多进程日志时，必须指定rank！"
            assert is_master is not None, "当使用多进程日志时，必须指定is_master！"
            self.rank = rank
            self.is_master = is_master
        else:
            self.rank = 0
            self.is_master = True

        self._make_logger()
        self._make_formatter()
        self._make_streamhandler()
        self._check_log_dir()
        self._make_filehandler()

        self.logger.info(
            f"logger初始化完成！\n"
            f"日志文件保存路径：{self.log_file}"
        )
    
    def _make_logger(self):
        self.logger = logging.getLogger(self.configs.project_name)
        self.logger.setLevel(logging.DEBUG)
    
    def _make_formatter(self):
        if self.is_multirank:
            self.formatter = logging.Formatter(
                (
                    f'[rank {self.rank}] '
                    '%(asctime)s | PID: %(process)d | "%(pathname)s", line %(lineno)d \n'
                    '| ------- %(levelname)s ------- |\n'
                    '%(message)s\n'
                ),
                '%Y-%m-%d %H:%M:%S'
            )
        else:
            self.formatter = logging.Formatter(
                (
                    '%(asctime)s | PID: %(process)d | "%(pathname)s", line %(lineno)d \n'
                    '| ------- %(levelname)s ------- |\n'
                    '%(message)s\n'
                ),
                '%Y-%m-%d %H:%M:%S'
            )
    
    def _make_streamhandler(self):
        if self.is_master:
            streamhandler = logging.StreamHandler()
            streamhandler.setLevel(logging.INFO)
            streamhandler.setFormatter(self.formatter)
            self.logger.addHandler(streamhandler)
    
    def _check_log_dir(self):
        log_dir_path = Path(self.configs.log_dir)
        if not log_dir_path.exists():
            log_dir_path.mkdir(exist_ok=True)
    
    def _make_filehandler(self):
        self.log_file = self.configs.log_dir + 'rank' + str(self.rank) + '.log'
        filehandler = logging.FileHandler(
            self.log_file, 
            'w', 
            encoding='utf-8'
        )
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(self.formatter)
        self.logger.addHandler(filehandler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, stacklevel=2, **kwargs)
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, stacklevel=2, **kwargs)
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, stacklevel=2, **kwargs)