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
    lyqdataset.download_from_hf_hub(
        LyqDataset.DatasetID.FINEWEB_EDU,
        batch_size=10000,
        batch_num=1000
    )