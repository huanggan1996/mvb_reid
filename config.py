from yacs.config import CfgNode

__C = CfgNode()
cfg = __C

__C.META_ARC = "mvb_pro"

__C.CUDA = True
__C.DEVICES = [0, 1]
# ------------------------------------------------------------------------ #
# Model params
# ------------------------------------------------------------------------ #
__C.MODEL = CfgNode()
__C.MODEL.GLOBAL_FEATS = 2048
__C.MODEL.PART_FEATS = 2048
# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CfgNode()
__C.TRAIN.DATA_ROOT = 'E:/data'
__C.TRAIN.DATASET = 'MVB'
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.NUM_IDENTITIES = 4
__C.TRAIN.NUM_EPOCHS = 150
__C.TRAIN.SNAPSHOT_DIR = './checkpoints'
__C.TRAIN.IMG_HEIGHT = 224
__C.TRAIN.IMG_WIDTH = 224
__C.TRAIN.MARGIN = 2
__C.TRAIN.EVAL_STEP = 10
__C.TRAIN.PRINT_FREQ = 40
__C.TRAIN.LOG_DIR = './logs'
__C.TRAIN.OPTIM = 'adam'
__C.TRAIN.XENT_LOSS_WEIGHT = 1
__C.TRAIN.TRIPLET_LOSS_WEIGHT = 1
__C.TRAIN.CENTER_LOSS_WEIGHT = 0.00025
# ------------------------------------------------------------------------ #
# Solver options
# ------------------------------------------------------------------------ #
__C.SOLVER = CfgNode()
__C.SOLVER.LEARNING_RATE = 0.00035
__C.SOLVER.LEARNING_RATE_CENTER = 0.5
__C.SOLVER.WEIGHT_DECAY = 5e-4
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.STEPS = [40, 70]
# ------------------------------------------------------------------------ #
# Testing options
# ------------------------------------------------------------------------ #
__C.TEST = CfgNode()
__C.TEST.OUTPUT = 'result.csv'