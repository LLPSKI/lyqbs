from typing import (
    TypedDict,
    TypeAlias
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

TraceSchema: TypeAlias = dict[str, TestTraceSchema]

def _process_trace_files(
    trace_files: list[Path],
    labels: list[str] | None = None
) -> TraceSchema:
    """

    """
    if labels is not None:
        assert len(trace_files) == len(labels), "长度不符！"
    else:
        labels = [i for i, _ in enumerate(trace_files)] # type: ignore
    
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
            'train_loss': []
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
        traces[labels[i]] = trace # type: ignore
    return traces

def draw_lab(
    trace_files: list[Path],
    begin_step: int = 0,
    end_step: int = -1,
) -> None:
    """
    指定一个trace.jsonl文件，绘制整体实验图
    """
    for trace_file in trace_files:
        assert trace_file.name == 'trace.jsonl', f"踪迹文件名称不符：{trace_file.name}！"

    assert True if end_step >= begin_step else True if end_step == -1 else False, f"开始步数{begin_step}和结束步数{end_step}不合法！"

    train_losses = []
    lrs = []
    tokens = []
    grad_norms = []
    for i, trace_file in enumerate(trace_files):
        
        train_losses.append([])
        lrs.append([])
        tokens.append([])
        grad_norms.append([])

        with trace_file.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                tracestep: TraceStepSchema = json.loads(line)

                train_losses[i].append(tracestep['train_loss'])
                lrs[i].append(tracestep['lr'])
                tokens[i].append(tracestep['tokens'])
                grad_norms[i].append(tracestep['grad_norm'])

    fig = plt.figure(
        num=None,
        figsize=(9, 12),
        dpi=300
    )

    ax_train_loss: Axes = fig.add_subplot(3, 1, 1)
    for i, y in enumerate(train_losses):
        ax_train_loss.plot(
            range(begin_step, end_step),
            y[begin_step:end_step],
            label=str(i)
        )
    ax_train_loss.set_xlabel("Step")
    ax_train_loss.set_ylabel("Loss")
    ax_train_loss.set_title("Training Loss Curve")

    ax_lr: Axes = fig.add_subplot(3, 1, 2)
    for y in lrs:
        ax_lr.plot(
            range(begin_step, end_step),
            y[begin_step:end_step]
        )
    ax_lr.set_xlabel("Step")
    ax_lr.set_ylabel("LR")

    ax_grad_norm: Axes = fig.add_subplot(3, 1, 3)
    for y in grad_norms:
        ax_grad_norm.plot(
            range(begin_step, end_step),
            y[begin_step:end_step]
        )
    fig.legend()
    fig.savefig('./train_loss.png')

if __name__ == '__main__':
    trace_files = [
        # Path('/mnt/hdd2/liuyuqi/output/Qwen/Qwen2.5-0.5B/2310e836a4a4fd16da39865bc611b019/trace.jsonl'),
        Path('/mnt/hdd2/liuyuqi/output/Qwen/Qwen2.5-0.5B/56c4a2bea38f750d351cc87c07768d3a/trace.jsonl'),
        # Path('/mnt/hdd2/liuyuqi/output/Qwen/Qwen2.5-0.5B/28c27c041ff847a010ca6ea55f18889e/trace.jsonl')
    ]
    draw_lab(
        trace_files,
        0,
        7500
    )
    # trace_file = Path('/mnt/hdd2/liuyuqi/output/Qwen/Qwen2.5-0.5B/56c4a2bea38f750d351cc87c07768d3a/trace.jsonl')

    # layers = []

    # with trace_file.open(encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         tracestep: TraceStepSchema = json.loads(line)

    #         if tracestep['layer_dict']:
    #             layers.append(tracestep['layer_dict'])
    #             continue
    
    # print(len(layers))
    # exit()

    # bucket_counts = layer_dict['module.model.embed_tokens.weight']
    # print(bucket_counts)
    # exit()

    # begin_exponent = 100
    # end_exponet = 127
    # target_layer = 4

    # fig = plt.figure(
    #     figsize=(12, 8),
    #     dpi=300
    # )

    # ax3d: Axes3D = fig.add_subplot(
    #     1, 1, 1,
    #     projection='3d'
    # )

    # for j, layer_dict in enumerate(layers):
    #     x = range(127)
    #     y = [j for _ in range(127)]
    #     it = iter(layer_dict.items())
    #     for _ in range(target_layer):
    #         k, v = next(it)
    #     ax3d.plot3D(
    #         x[begin_exponent:end_exponet],
    #         y[begin_exponent:end_exponet],
    #         v[begin_exponent:end_exponet]
    #     )

    # for i, (k, v) in enumerate(layer_dict.items()):
    #     if i == 0:
    #         continue
    #     x = range(127)
    #     y = [i for _ in range(127)]

    #     ax3d.plot3D(
    #         x[begin_exponent:end_exponet],
    #         y[begin_exponent:end_exponet],
    #         v[begin_exponent:end_exponet]
    #     )

    # ax3d.view_init(elev=20, azim=45)

    # fig.savefig('./distribution.png')