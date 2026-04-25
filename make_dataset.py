from lyq.utils.data import (
    LyqDataset
)
from lyq.config import Configs
from lyq.log import Logger

if __name__ == '__main__':
    configs = Configs()
    logger = Logger(
        configs,
    )
    lyqdataset = LyqDataset(
        configs,
        logger
    )

    # 注意：不要一次下载太多，可能会超时报错
    lyqdataset.download_from_hf_hub(
        LyqDataset.DatasetID.FINEWEB_EDU,
        batch_size=10000,
        batch_num=64
    )

    is_valid = lyqdataset.verify()
    if is_valid:
        logger.info("数据集下载成功！")
    else:
        logger.info("数据集下载失败！！！")