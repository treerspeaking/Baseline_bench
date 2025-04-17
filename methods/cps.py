import argparse

import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torchmetrics import Accuracy
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np
import yaml

from data import (
    MySVHN, MyCifar10
)
from utils.utils import net_factory
from utils.ramps import sigmoid_ramp_up, cosine_ramp_down

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
# parser.add_argument('--labeled-id-path', type=str, required=True)
# parser.add_argument('--unlabeled-id-path', type=str, required=True)
# parser.add_argument('--save-path', type=str, required=True)
# parser.add_argument('--local_rank', default=0, type=int)
# parser.add_argument('--port', default=None, type=int)

args = parser.parse_args()
cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

class CPS(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model_1 = net_factory(cfg["network"], cfg["in_channels"], cfg["num_classes"])
        self.model_2 = net_factory(cfg["network"], cfg["in_channels"], cfg["num_classes"])
        
    def training_step(self, batch, batch_idx):
        
        labeled_data_s, labeled_data_t, label = batch[0]
        unlabeled_data_s, unlabeled_data_t = batch[1]
        labeled_data_len = len(labeled_data_s)
        unlabeled_data_len = len(unlabeled_data_s)
        mini_batch = labeled_data_len + unlabeled_data_len
        
        combine_data_s = torch.concat((labeled_data_s, unlabeled_data_s), dim=0)
        combine_data_t = torch.concat((labeled_data_t, unlabeled_data_t), dim=0)
        
        stu_out = self.student(combine_data_s)
        tea_out = self.teacher(combine_data_t)