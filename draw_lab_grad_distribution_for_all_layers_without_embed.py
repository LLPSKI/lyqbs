from pathlib import Path

from lyq import *
from lyq.utils.trace import (
    draw_lab_grad_distribution_for_all_layers
)

if __name__ == '__main__':
    configs = Configs()

    # 替换成所需实验的trace_file
    trace_file: Path = Path('/mnt/hdd2/liuyuqi/output/Qwen/Qwen2.5-0.5B/d40248a9c382ea1d54aa6f671465407a/trace.jsonl')
    
    draw_lab_grad_distribution_for_all_layers(
        configs.image_dir,
        trace_file,
        without_embed=True
    )