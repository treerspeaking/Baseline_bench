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

from utils.data import (
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
        
        stu_out = self.model_1(combine_data_s)
        tea_out = self.model_2(combine_data_t)
        
    def validation_step(self, batch, batch_idx):
        
        labeled_data, labels = batch
        mini_batch = len(labeled_data)
        stu_out = self.student(labeled_data)
        tea_out = self.teacher(labeled_data)
        
                
        if isinstance(stu_out, tuple):
            stu_out, stu_out_2 = stu_out
            tea_out, _ = tea_out
            loss_logit = self.logit_loss(stu_out, stu_out_2) / self.num_classes / mini_batch
        else:
            loss_logit = 0
        
        loss_ce_stu = self.ce_loss(stu_out, labels) / mini_batch
        loss_ce_tea = self.ce_loss(tea_out, labels) / mini_batch 
        
        loss_cons = self.mse_loss(F.softmax(stu_out, dim=1), F.softmax(tea_out, dim=1)) / self.num_classes / mini_batch * self.consistency_weight * self.ramp_up(self.global_step)

        
        stu_acc = self.acc(stu_out, labels) 
        tea_acc = self.acc(tea_out, labels) 
        
        # for i in range(self.num_classes):
        #     log_dict_preds[f"val_stu_pred_class_{i}"] = (stu_preds == i).sum().item()
        #     log_dict_preds[f"val_tea_pred_class_{i}"] = (tea_preds == i).sum().item()
            
        # self.log_dict(log_dict_preds, on_epoch=True, on_step=False) # Log prediction counts per epoch
        
        
        
        self.log_dict({
            "val_loss_ce_stu": loss_ce_stu, 
            "val_loss_ce_tea": loss_ce_tea,
            "val_loss_cons": loss_cons,
            "val_loss_logit": loss_logit,
            "val_total": (loss_cons + loss_ce_tea + loss_logit) ,
            "val_stu_acc": stu_acc,
            "val_tea_acc": tea_acc
        }, on_epoch=True, on_step=True)
        
    def ramp_up(self, steps):
        return sigmoid_ramp_up(steps, self.ramp_up_length)
    
    def ramp_lr(self, steps):
        return cosine_ramp_down(steps, self.ramp_down_length)
        
        
    # def on_before_optimizer_step(self, optimizer):
    #     # Change Adam's beta2 parameter after ramp up
    #     if self.global_step == self.ramp_up_length:
    #         for param_group in optimizer.param_groups:
    #             current_beta1 = param_group['betas'][0]
    #             new_beta2 = cfg["adam_beta_2_after_ramp_up"] # New beta2 value
    #             param_group['betas'] = (current_beta1, new_beta2)
    #             self.log("optimizer_beta2_changed", 1.0)
    #             print(f"Updated optimizer beta2 to {new_beta2} at step {self.global_step}")

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=cfg["lr"], betas=(cfg["adam_beta_1"], cfg["adam_beta_2_during_ramp_up"]), eps=1e-8)
        optimizer = torch.optim.SGD(self.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay,
                                nesterov=self.nesterov)
        # optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, betas=(0.9, 0.99), eps=1e-8)
        scheduler = LambdaLR(optimizer, self.ramp_lr)
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

trainer = L.Trainer(
    max_steps=cfg["steps"], 
    enable_checkpointing=True, 
    log_every_n_steps=100, 
    check_val_every_n_epoch=None, 
    val_check_interval=cfg["val_check_interval"],
    accumulate_grad_batches = 1
    )
trainer.fit(model=CPS(), train_dataloaders=train_loader, val_dataloaders=eval_loader)