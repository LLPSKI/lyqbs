from lyq import *
from lyq.dist import *
from lyq.dist.lab import *

if __name__ == '__main__':
    with global_env():
        configs = Configs()
        logger = Logger(
            configs,
            is_multirank=True,
            rank=rank(),
            is_master=is_master()
        )
        lab = LyqLab(
            configs,
            logger,
            quan=LyqLab.Quan.S1E1M6_111_QUAN
        )
        
        lab.train(
            checkpoint_nums=4,
            checkpoint_steps=10000
        )

        # 测试模型训练有效性需要较长的时间
        # if not lab.verify():
        #     logger.info("实验训练好像出了什么问题？！")
        # else:
        #     logger.info("实验训练没有问题！")