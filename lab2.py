from lyq import *

if __name__ == '__main__':
    with global_env():
        lab2 = LyqLab(
            optim_configs = {
                'lr': 5e-05,
                "betas": (0.9, 0.95),
                "eps": 1e-08,
                "weight_decay": 0.1
            },
            lr=LyqLab.LR.LWLDLR,
            lr_configs = {
                "warmup_ratio": 0.05,
                "min_lr_ratio": 0.05
            },
            total_steps=10000
        )
        lab2.train(
            checkpoint_nums=5,
            checkpoint_steps=500
        )