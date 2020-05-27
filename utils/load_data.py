import logging
from datasets.data_loader import MyDataset
from datasets.data_manager import init_dataset
from datasets.samplers import RandomIdentitySampler
from torch.utils.data import DataLoader
from utils.transforms import TrainTransformer, TestTransformer
from config import cfg
logger = logging.getLogger('global')


def build_data_loader():
    logger.info("build train dataset")
    # dataset
    dataset = init_dataset(cfg.TRAIN.DATA_ROOT, cfg.TRAIN.DATASET)
    sampler = RandomIdentitySampler(dataset.train, cfg.TRAIN.NUM_IDENTITIES)
    train_loader = DataLoader(MyDataset(dataset.train, TrainTransformer()),
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=4,
                              pin_memory=True,
                              sampler=sampler)

    query_loader = DataLoader(MyDataset(dataset.query, TestTransformer()),
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=4,
                              pin_memory=True,
                              shuffle=False)

    gallery_loader = DataLoader(MyDataset(dataset.gallery, TestTransformer()),
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=False)
    return dataset, train_loader, query_loader, gallery_loader
