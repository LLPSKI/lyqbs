from lyq import *

if __name__ == '__main__':
    with global_env():
        LyqLab().train(
            checkpoint_nums=1,
            checkpoint_steps=100
        )