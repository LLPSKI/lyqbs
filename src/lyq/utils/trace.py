from typing import (
    TypedDict,
    TypeAlias,
    Iterable
)
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.axes import (
    Axes,
)
from mpl_toolkits.mplot3d import (
    Axes3D
)

__all__ = [
    "draw_lab_trace_view",
    "draw_lab_grad_distribution_for_all_layers",
    "draw_lab_grad_distribution_for_given_layer"
]

class TraceStepSchema(TypedDict):
    """
    step: 当前步数，从0开始  
      
    超参数：  
    lr: 当前步全局学习率
      
    时间：  
    training_time: 当前步训练总时间  
    grad_sync_total_time: 当前步梯度同步总时间  
    grad_sync_comp_time: 通信前预处理时间  
    grad_sync_comm_time: 通信时间  
      
    通信量：  
    grad_sync_comm_bytes: 所有计算节点所需要通信的总字节数
      
    指标：  
    tokens: 当前步所有计算节点总共吃到的有效词数量  
    train_loss: 当前步训练损失  
    grad_norm: 当前步梯度范数  
      
    梯度分布：  
    layer_dict: 参数张量的bucket_counts字典，值为列表对象，长度为127，记录归一化梯度值依据阶码分桶
    """
    step: int

    lr: float

    train_time: float
    grad_sync_total_time: float
    grad_sync_comp_time: float
    grad_sync_comm_time: float

    grad_sync_comm_bytes: int

    tokens: int
    train_loss: float
    grad_norm: float

    layer_dict: dict[str, list[int]]

class TestTraceSchema(TypedDict):
    lr: list[float]
    train_loss: list[float]
    tokens: list[int]
    grad_norm: list[float]
    grad_sync_comm_bytes: list[int]
    grad_sync_total_time: list[float]
    grad_sync_comp_time: list[float]
    grad_sync_comm_time: list[float]
    layer_dict: list[dict[str, list[int]]]

TraceSchema: TypeAlias = dict[str, TestTraceSchema]

def _process_trace_files(
    trace_files: list[Path],
    labels: list[str] | None = None
) -> TraceSchema:
    """

    """
    if labels is not None:
        assert len(trace_files) == len(labels), "长度不符！"
    
    traces: TraceSchema = {}

    for i, trace_file in enumerate(trace_files):
        trace: TestTraceSchema = {
            'grad_norm': [],
            'grad_sync_comm_bytes': [],
            'grad_sync_comm_time': [],
            'grad_sync_comp_time': [],
            'grad_sync_total_time': [],
            'lr': [],
            'tokens': [],
            'train_loss': [],
            'layer_dict': []
        }
        with trace_file.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                tracestep: TraceStepSchema = json.loads(line)

                trace['grad_norm'].append(tracestep['grad_norm'])
                trace['grad_sync_comm_bytes'].append(tracestep['grad_sync_comm_bytes'])
                trace['grad_sync_comm_time'].append(tracestep['grad_sync_comm_time'])
                trace['grad_sync_comp_time'].append(tracestep['grad_sync_comp_time'])
                trace['grad_sync_total_time'].append(tracestep['grad_sync_total_time'])
                trace['lr'].append(tracestep['lr'])
                trace['tokens'].append(tracestep['tokens'])
                trace['train_loss'].append(tracestep['train_loss'])
                trace['layer_dict'].append(tracestep['layer_dict'])
        if labels is not None:
            traces[labels[i]] = trace # type: ignore
        else:
            traces[trace_file.parent.name] = trace

    return traces

def _draw_loss_lines(
    axes: Axes,
    ys: Iterable[Iterable],
    begin_step: int,
    end_step: int,
    labels: list[str]
) -> None:
    """
    画loss曲线
    """
    for y, label in zip(ys, labels):
        axes.plot(
            range(begin_step, end_step), 
            y[begin_step:end_step], # type: ignore
            linewidth=1.5,
            label=label
        )
    axes.set_xlabel("Step")
    axes.set_ylabel("Train Loss")
    axes.set_title("Train Loss over Steps")
    axes.grid(True)

    axes.legend(
        shadow=True,
        fontsize=12
    )

def _draw_loss_line(
    axes: Axes,
    y: Iterable,
    begin_step: int,
    end_step: int,
) -> None:
    """
    画loss曲线
    """
    axes.plot(
        range(begin_step, end_step), 
        y[begin_step:end_step], # type: ignore
        linewidth=1.5
    )
    axes.set_xlabel("Step")
    axes.set_ylabel("Train Loss")
    axes.set_title("Train Loss over Steps")
    axes.grid(True)

def _draw_lr_lines(
    axes: Axes,
    ys: Iterable[Iterable],
    begin_step: int,
    end_step: int,
    labels: list[str]
) -> None:
    """
    画lr曲线
    """
    for y, label in zip(ys, labels):
        axes.plot(
            range(begin_step, end_step), 
            y[begin_step:end_step], # type: ignore
            linewidth=1.5,
            label=label
        )
    axes.set_xlabel("Step")
    axes.set_ylabel("LR")
    axes.set_title("LR over Steps")
    axes.grid(True)

    axes.legend(
        shadow=True,
        fontsize=12
    )

def _draw_lr_line(
    axes: Axes,
    y: Iterable,
    begin_step: int,
    end_step: int,
) -> None:
    """
    画lr曲线
    """
    axes.plot(
        range(begin_step, end_step), 
        y[begin_step:end_step], # type: ignore
        linewidth=1.5
    )
    axes.set_xlabel("Step")
    axes.set_ylabel("LR")
    axes.set_title("LR over Steps")
    axes.grid(True)

def _draw_grad_norm_lines(
    axes: Axes,
    ys: Iterable[Iterable],
    begin_step: int,
    end_step: int,
    labels: list[str]
) -> None:
    """
    画grad_norm曲线
    """
    for y, label in zip(ys, labels):
        axes.plot(
            range(begin_step, end_step), 
            y[begin_step:end_step], # type: ignore
            linewidth=1.5,
            label=label
        )
    axes.set_xlabel("Step")
    axes.set_ylabel("Grad Norm")
    axes.set_title("Grad Norm over Steps")
    axes.grid(True)

def _draw_grad_norm_line(
    axes: Axes,
    y: Iterable,
    begin_step: int,
    end_step: int,
) -> None:
    """
    画grad_norm曲线
    """
    axes.plot(
        range(begin_step, end_step), 
        y[begin_step:end_step], # type: ignore
        linewidth=1.5
    )
    axes.set_xlabel("Step")
    axes.set_ylabel("Grad Norm")
    axes.set_title("Grad Norm over Steps")
    axes.grid(True)

def _draw_tokens_line(
    axes: Axes,
    y: Iterable,
    begin_step: int,
    end_step: int,
) -> None:
    """
    画tokens曲线
    """
    axes.plot(
        range(begin_step, end_step), 
        y[begin_step:end_step], # type: ignore
        linewidth=1.5
    )
    avg = sum(y[begin_step:end_step]) / (end_step - begin_step) # type: ignore
    axes.text(
        0.95, 
        0.95, 
        f"Avg Tokens per Step: {int(avg)}",
        transform=axes.transAxes,
        ha='right',
        va='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor='gray',
        )
    )
    axes.set_xlabel("Step")
    axes.set_ylabel("Tokens")
    axes.set_title("Tokens over Steps")
    axes.grid(True)

def _draw_grad_sync_time_line(
    axes: Axes,
    total: Iterable,
    comm: Iterable,
    begin_step: int,
    end_step: int,
) -> None:
    """
    画通信时间占比图
    """
    axes.plot(
        range(begin_step, end_step), 
        total[begin_step:end_step], # type: ignore
        linewidth=1.5,
        label='total time'
    )
    axes.plot(
        range(begin_step, end_step), 
        comm[begin_step:end_step], # type: ignore
        linewidth=1.5,
        label='comm time'
    )
    axes.set_xlabel("Step")
    axes.set_ylabel("Time")
    axes.set_title("Grad Sync Time over Steps")
    axes.grid(True)

    axes.legend(
        shadow=True,
        fontsize=12
    )

def _draw_comm_bytes_line(
    axes: Axes,
    y: Iterable,
    begin_step: int,
    end_step: int,
) -> None:
    """
    画通信量曲线
    """
    axes.plot(
        range(begin_step, end_step), 
        y[begin_step:end_step], # type: ignore
        linewidth=1.5
    )
    avg = sum(y[begin_step:end_step]) / (end_step - begin_step) # type: ignore
    axes.text(
        0.95, 
        0.95, 
        f"Avg Comm Bytes per Step: {avg / 1024 / 1024:.4f}MB",
        transform=axes.transAxes,
        ha='right',
        va='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor='gray',
        )
    )
    axes.set_xlabel("Step")
    axes.set_ylabel("Comm bytes")
    axes.set_title("Comm bytes over Steps")
    axes.grid(True)

def draw_lab_trace_view(
    image_dir: str,
    trace_file: Path,
    begin_step: int = 0,
    end_step: int = 100,
) -> None:
    """
    指定trace.jsonl文件，绘制整体实验图
    """
    assert trace_file.name == 'trace.jsonl', f"踪迹文件名称不符：{trace_file.name}！"

    assert True if end_step >= begin_step else False, f"开始步数{begin_step}和结束步数{end_step}不合法！"

    traces = _process_trace_files(
        [trace_file]
    )

    fig = plt.figure(
        num=None,
        figsize=(15, 15),
        dpi=300
    )

    ax_train_loss: Axes = fig.add_subplot(3, 2, 1)
    _draw_loss_line(
        ax_train_loss,
        list(traces.values())[0]['train_loss'],
        begin_step,
        end_step,
    )

    ax_tokens: Axes = fig.add_subplot(3, 2, 2)
    _draw_tokens_line(
        ax_tokens,
        list(traces.values())[0]['tokens'],
        begin_step,
        end_step,
    )

    ax_lr: Axes = fig.add_subplot(3, 2, 3)
    _draw_lr_line(
        ax_lr,
        list(traces.values())[0]['lr'],
        begin_step,
        end_step
    )

    ax_grad_sync_time: Axes = fig.add_subplot(3, 2, 4)
    _draw_grad_sync_time_line(
        ax_grad_sync_time,
        list(traces.values())[0]['grad_sync_total_time'],
        list(traces.values())[0]['grad_sync_comm_time'],
        begin_step,
        end_step
    )

    ax_grad_norm: Axes = fig.add_subplot(3, 2, 5)
    _draw_grad_norm_line(
        ax_grad_norm,
        list(traces.values())[0]['grad_norm'],
        begin_step,
        end_step
    )

    ax_comm_bytes: Axes = fig.add_subplot(3, 2, 6)
    _draw_comm_bytes_line(
        ax_comm_bytes,
        list(traces.values())[0]['grad_sync_comm_bytes'],
        begin_step,
        end_step
    )

    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.05,
        top=0.9,
        wspace=0.25,
        hspace=0.2
    )

    fig.suptitle(
        f'Lab: {trace_file.parent.name} Trace View',
        fontsize=18
    )

    image_dir_path = Path(image_dir) / trace_file.parent.name
    if not image_dir_path.exists():
        image_dir_path.mkdir(
            parents=True,
            exist_ok=True
        )
    fig.savefig(image_dir_path / 'train_loss.png')

def draw_lab_grad_distribution_for_all_layers(
    image_dir: str,
    trace_file: Path,
    begin_step: int,
    end_step: int,
    without_embed: bool = False,
    begin_exponent: int = 0,
    end_exponent: int = 127,
) -> None:
    """
    指定trace.jsonl文件，将整个模型作为一个整体绘制梯度分布图
    """
    assert trace_file.name == 'trace.jsonl', f"踪迹文件名称不符：{trace_file.name}！"

    traces = _process_trace_files(
        [trace_file]
    )

    fig = plt.figure(
        num=None,
        figsize=(8, 8),
        dpi=300
    )

    ax3d: Axes3D = fig.add_subplot(
        1, 1, 1,
        projection='3d'
    )
    
    for j, layer_dict in enumerate(list(traces.values())[0]['layer_dict']):
        if not layer_dict or j < begin_step or j > end_step:
            continue
        bucket_counts = [0 for _ in range(127)]
        for i, (name, layer) in enumerate(layer_dict.items()):
            if without_embed and i == 0:
                continue
            bucket_counts = [x + y for x, y in zip(bucket_counts, layer)]
        x = range(127)
        y = [j for _ in range(127)]
        ax3d.plot3D(
            x[begin_exponent:end_exponent],
            y[begin_exponent:end_exponent],
            bucket_counts[begin_exponent:end_exponent]
        )

    ax3d.view_init(elev=20, azim=90)

    image_dir_path = Path(image_dir) / trace_file.parent.name
    if not image_dir_path.exists():
        image_dir_path.mkdir(
            parents=True,
            exist_ok=True
        )
    if without_embed:
        fig.savefig(image_dir_path / 'all_layers_without_embed_grad_distribution.png')
    else:
        fig.savefig(image_dir_path / 'all_layers_grad_distribution.png')

def draw_lab_grad_distribution_for_given_layer(
    image_dir: str,
    trace_file: Path,
    layer: int,
    begin_exponent: int = 0,
    end_exponent: int = 127,
) -> None:
    """
    指定trace.jsonl文件，将给定层作为一个整体绘制梯度分布图
    """
    assert trace_file.name == 'trace.jsonl', f"踪迹文件名称不符：{trace_file.name}！"

    traces = _process_trace_files(
        [trace_file]
    )

    fig = plt.figure(
        num=None,
        figsize=(8, 8),
        dpi=300
    )

    ax3d: Axes3D = fig.add_subplot(
        1, 1, 1,
        projection='3d'
    )
    
    for j, layer_dict in enumerate(list(traces.values())[0]['layer_dict']):
        if not layer_dict:
            continue
        layer_name = list(layer_dict.keys())[layer]
        bucket_counts = layer_dict[layer_name]
        x = range(127)
        y = [j for _ in range(127)]
        ax3d.plot3D(
            x[begin_exponent:end_exponent],
            y[begin_exponent:end_exponent],
            bucket_counts[begin_exponent:end_exponent]
        )

    ax3d.view_init(elev=20, azim=90)

    image_dir_path = Path(image_dir) / trace_file.parent.name
    if not image_dir_path.exists():
        image_dir_path.mkdir(exist_ok=True)
    fig.savefig(image_dir_path / f'{layer_name}: grad_distribution.png')