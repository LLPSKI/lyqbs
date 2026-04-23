import logging
from pathlib import Path
from .config import *
from .env import *

class _Logger:
    """
    项目日志类
    """
    def __init__(self) -> None:
        """
        设置全局统一的日志对象  
        每个进程有一个独立的日志对象，不同的日志文件
        主进程具有打印到终端的能力，其他进程只能写入到日志文件中
        """
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
        self.logger = logging.getLogger(project_name())
        self.logger.setLevel(logging.DEBUG)
    
    def _make_formatter(self):
        self.formatter = logging.Formatter(
            (
                f'[rank {rank()}] '
                '%(asctime)s | PID: %(process)d | "%(pathname)s", line %(lineno)d \n'
                '| ------- %(levelname)s ------- |\n'
                '%(message)s\n'
            ),
            '%Y-%m-%d %H:%M:%S'
        )
    
    @master_function
    def _make_streamhandler(self):
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(logging.INFO)
        streamhandler.setFormatter(self.formatter)
        self.logger.addHandler(streamhandler)
    
    def _check_log_dir(self):
        log_dir_path = Path(log_dir())
        if not log_dir_path.exists():
            log_dir_path.mkdir(exist_ok=True)
    
    def _make_filehandler(self):
        self.log_file = log_dir() + 'rank' + str(rank()) + '.log'
        filehandler = logging.FileHandler(
            self.log_file, 
            'w', 
            encoding='utf-8'
        )
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(self.formatter)
        self.logger.addHandler(filehandler)

_l = _Logger()

def debug(msg, *args, **kwargs):
    _l.logger.debug(msg, *args, stacklevel=2, **kwargs)
def info(msg, *args, **kwargs):
    _l.logger.info(msg, *args, stacklevel=2, **kwargs)
def error(msg, *args, **kwargs):
    _l.logger.error(msg, *args, stacklevel=2, **kwargs)