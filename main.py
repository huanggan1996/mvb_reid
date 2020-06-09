import logging
import shutil
import time
import torch
import os

from tensorboardX import SummaryWriter
from config import cfg

from models.network import BagReID_RESNET, BagReID_SE_RESNEXT, BagReID_IBN
from utils.loss import TripletLoss, CrossEntropyLabelSmooth, CenterLoss
from utils.log_helper import init_log, add_file_handler
from utils.load_data import build_data_loader
from utils.lr_scheduler import WarmupMultiStepLR
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint

logger = logging.getLogger('global')

criterion_xent = None
criterion_triplet = None
criterion_center = None


def criterion(logits, features, ids):
    global criterion_xent, criterion_triplet, criterion_center
    xcent_loss = sum([criterion_xent(output, ids) for output in logits])
    triplet_loss = sum([criterion_triplet(output, ids)[0] for output in features])
    center_loss = criterion_center(torch.cat(features, dim=1), ids)
    loss = cfg.TRAIN.XENT_LOSS_WEIGHT * xcent_loss + \
           cfg.TRAIN.TRIPLET_LOSS_WEIGHT * triplet_loss + \
           cfg.TRAIN.CENTER_LOSS_WEIGHT * center_loss
    return loss


def train(epoch, train_loader, model, criterion, optimizers, summary_writer):
    global criterion_center
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR):
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    # start training
    model.train()
    start = time.time()
    for ii, datas in enumerate(train_loader):
        data_time.update(time.time() - start)
        img, bag_id, cam_id = datas
        if cfg.CUDA:
            img = img.cuda()
            bag_id = bag_id.cuda()

        triplet_features, softmax_features = model(img)

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = criterion(softmax_features, triplet_features, bag_id)
        loss.backward()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_center.parameters():
            param.grad.data *= (1. / cfg.TRAIN.CENTER_LOSS_WEIGHT)

        for optimizer in optimizers:
            optimizer.step()

        batch_time.update(time.time() - start)
        losses.update(loss.item())
        # tensorboard
        if summary_writer:
            global_step = epoch * len(train_loader) + ii
            summary_writer.add_scalar('loss', loss.item(), global_step)

        start = time.time()

        if (ii + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            logger.info('Epoch: [{}][{}/{}]\t'
                        'Batch Time {:.3f} ({:.3f})\t'
                        'Data Time {:.3f} ({:.3f})\t'
                        'Loss {:.3f} ({:.3f}) \t'
                        .format(epoch + 1, ii + 1, len(train_loader),
                                batch_time.val, batch_time.mean,
                                data_time.val, data_time.mean,
                                losses.val, losses.mean))
    adam_param_groups = optimizers[0].param_groups
    logger.info('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
                'Adam Lr {:.2e} \t '
                .format(epoch + 1, batch_time.sum, losses.mean,
                        adam_param_groups[0]['lr']))


def build_lr_schedulers(optimizers):
    schedulers = []
    for optimizer in optimizers:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS)
        schedulers.append(scheduler)
    return schedulers


def main():
    logger = logging.getLogger('global')
    global criterion_xent, criterion_triplet, criterion_center
    if os.path.exists(cfg.TRAIN.LOG_DIR):
        shutil.rmtree(cfg.TRAIN.LOG_DIR)
    os.makedirs(cfg.TRAIN.LOG_DIR)
    init_log('global', logging.INFO)  # log
    add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)
    summary_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)  # visualise

    dataset, train_loader, _, _ = build_data_loader()
    model = BagReID_RESNET(dataset.num_train_bags)
    criterion_xent = CrossEntropyLabelSmooth(dataset.num_train_bags, use_gpu=cfg.CUDA)
    criterion_triplet = TripletLoss(margin=cfg.TRAIN.MARGIN)
    criterion_center = CenterLoss(dataset.num_train_bags,
                                  cfg.MODEL.GLOBAL_FEATS + cfg.MODEL.PART_FEATS, use_gpu=cfg.CUDA)
    if cfg.TRAIN.OPTIM == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.SOLVER.LEARNING_RATE,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.SOLVER.LEARNING_RATE,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    center_optimizer = torch.optim.SGD(criterion_center.parameters(),
                                       lr=cfg.SOLVER.LEARNING_RATE_CENTER)

    optimizers = [optimizer, center_optimizer]
    schedulers = build_lr_schedulers(optimizers)

    if cfg.CUDA:
        model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=cfg.DEVICES)

    logger.info("model prepare done")
    # start training
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        train(epoch, train_loader, model, criterion, optimizers, summary_writer)
        for scheduler in schedulers:
            scheduler.step()

        # skip if not save model
        if cfg.TRAIN.EVAL_STEP > 0 and (epoch + 1) % cfg.TRAIN.EVAL_STEP == 0 \
                or (epoch + 1) == cfg.TRAIN.NUM_EPOCHS:

            if cfg.CUDA and torch.cuda.device_count() > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
                            is_best=False, save_dir=cfg.TRAIN.SNAPSHOT_DIR,
                            filename='checkpoint_ep' + str(epoch + 1) + '.pth')


if __name__ == '__main__':
    main()
