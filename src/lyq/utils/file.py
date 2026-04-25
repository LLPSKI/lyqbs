from pathlib import Path

__all__ = [
    'is_files_all_exists',
    'find_max_checkpoint',
    'find_all_file'
]

def is_files_all_exists(
    dir: str, 
    files: list[str]
) -> bool:
    """
    检查某一文件夹下是否具有全部要求的文件
    """
    is_all_exists = True
    files = [dir + file for file in files]
    for file in files:
        file = Path(file)
        if not file.exists():
            is_all_exists = False
            break
    return is_all_exists

def find_max_checkpoint(
    dir: str
) -> Path | None:
    """
    检查某一目录下的保存点文件夹
    """
    path = Path(dir)
    checkpoints = [
        p for p in path.glob("checkpoint-*")
        if p.is_dir()
    ]

    if not checkpoints:
        return None
    
    return max(checkpoints, key=lambda x: x.name)

def find_all_file(
    dir: str,
    target: str
) -> list[Path]:
    """
    得到指定目录下所有的目标文件
    递归查找，没有找到则返回空列表
    """
    path = Path(dir)
    target_files = [
        p for p in path.rglob(target)
        if p.is_file()
    ]

    return target_files