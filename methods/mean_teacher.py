import argparse

import lightning as L
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torchmetrics import Accuracy
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
import yaml

from utils.data import (
    MySVHN, MyCifar10
)
from utils.utils import net_factory
from utils import data, losses


from utils.ramps import sigmoid_ramp_up, cosine_ramp_down

NO_LABEL = -1
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
# parser.add_argument('--labeled-id-path', type=str, required=True)
# parser.add_argument('--unlabeled-id-path', type=str, required=True)
# parser.add_argument('--save-path', type=str, required=True)
# parser.add_argument('--local_rank', default=0, type=int)
# parser.add_argument('--port', default=None, type=int)

args = parser.parse_args()
cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

class MeanTeacher(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.student = net_factory(cfg["network"], cfg["in_channels"], cfg["num_classes"])
        self.teacher = net_factory(cfg["network"], cfg["in_channels"], cfg["num_classes"])
        for param in self.teacher.parameters():
                param.detach_()
        self.num_classes = cfg["num_classes"]
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL, size_average=False)
        self.mse_loss = torch.nn.MSELoss(size_average=False)
        self.logit_loss = torch.nn.MSELoss(size_average=False)
        self.acc = Accuracy('multiclass', num_classes=cfg["num_classes"])
        self.consistency_weight = cfg["consistency_weight"]
        self.residual_weight = cfg["residual_weight"]
        self.ramp_up_length = cfg["ramp_up_length"]
        self.ramp_down_length = cfg["ramp_down_length"]
        self.ema_decay_during_ramp_up = cfg["ema_decay_during_ramp_up"]
        self.ema_decay_after_ramp_down = cfg["ema_decay_after_ramp_up"]
        self.lr = torch.tensor(cfg["lr"])
        self.momentum=torch.tensor(cfg["momentum"])
        self.weight_decay=torch.tensor(cfg["weight_decay"])
        self.nesterov=torch.tensor(cfg["nesterov"])
        self.save_hyperparameters()
        
    def ema(self, student, teacher, alpha):
        
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        with torch.no_grad():
            # Update parameters
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(alpha).add_((1 - alpha) * s_param.data)
            
            # Update only floating-point buffers
            # for t_buffer, s_buffer in zip(teacher.buffers(), student.buffers()):
            #     if t_buffer.dtype.is_floating_point:
            #         t_buffer.data.mul_(alpha).add_((1 - alpha) * s_buffer.data)
            # Non-floating-point buffers (e.g., num_batches_tracked) are left unchanged
            
        
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
        
        if isinstance(stu_out, tuple):
            stu_out, stu_out_cons = stu_out
            tea_out, _ = tea_out
            loss_logit = self.logit_loss(stu_out, stu_out_cons) / self.num_classes / mini_batch * self.residual_weight
        else:
            loss_logit = 0 

        loss_l = self.ce_loss(stu_out[:labeled_data_len], label) / mini_batch
        loss_u = self.mse_loss(F.softmax(stu_out_cons, dim=1), F.softmax(tea_out, dim=1)) / self.num_classes / mini_batch * self.consistency_weight * self.ramp_up(self.current_epoch)
        
        # Calculate accuracy for the labeled part
        train_acc = self.acc(stu_out[:labeled_data_len], label)
        self.log("train_acc", train_acc, on_step=True, on_epoch=False) # Log training accuracy per step
        
        
        self.log_dict({
            "consistency_weight": self.consistency_weight * self.ramp_up(self.current_epoch),
            "train_loss_l": loss_l,
            "train_loss_u": loss_u,
            "train_loss_logit": loss_logit,
            "learning_rate": self.trainer.optimizers[0].param_groups[0]['lr'],
            "total_loss": (loss_l + loss_u + loss_logit)  # this is mean cost to be minimize (yes it is loss_l + loss_u / 3)
        })
        return (loss_l + loss_u + loss_logit) 
        
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        
        if self.global_step <= self.ramp_up_length:
            self.ema(self.student, self.teacher, self.ema_decay_during_ramp_up)
        else:
            self.ema(self.student, self.teacher, self.ema_decay_after_ramp_down)
        
        return 
    
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
        optimizer = torch.optim.SGD(self.student.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay,
                                nesterov=self.nesterov)
        # optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, betas=(0.9, 0.99), eps=1e-8)
        scheduler = LambdaLR(optimizer, self.ramp_lr)
        
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            # "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

cifar10_train = MyCifar10(True, transforms=v2.Compose([
    # data.RandomTranslateWithReflect(4),
    # v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616]),
    v2.RandomAffine(degrees=0, translate=(0.126, 0.126)),
    v2.RandomHorizontalFlip()
]))

Cifar10_test = MyCifar10(False, transforms=v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616]),
]))

labeled_dataloader, unlabeled_dataloader = cifar10_train.get_dataloader(labeled_batch_size=cfg["labeled_batch_size"], labeled_num_worker=4,shuffle = True, unlabeled=True, labeled_size=cfg["labeled_sample"], unlabeled_batch_size=cfg["unlabeled_batch_size"], unlabeled_num_worker = 4, seed= None)
test_dataloader = Cifar10_test.get_dataloader(200, 4, False)

# SVHN_train = MySVHN(".", split="train", transforms=v2.Compose([
#     v2.ToTensor(),
#     v2.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
#     # v2.RandomHorizontalFlip(), # not applied on svhn !??
#     v2.RandomAffine(degrees=0, translate=(0.0626, 0.0626)),
#     v2.GaussianNoise(sigma=0.15)
# ]), download=True)

# SVHN_test = MySVHN(".", split="test", transforms=v2.Compose([
#     v2.ToTensor(),
#     v2.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
# ]), download=True)


# labeled_dataloader, unlabeled_dataloader = SVHN_train.get_dataloader(labeled_batch_size=cfg.labeled_batch_size, labeled_num_worker=4,shuffle = True, unlabeled=True, labeled_size=cfg.labeled_sample, unlabeled_batch_size=cfg.unlabeled_batch_size, unlabeled_num_worker = 4, seed= None)
# test_dataloader = SVHN_test.get_dataloader(200, 4, False)

# print("number labeled:", len(labeled_dataloader))
# print("number unlabeld:", len(unlabeled_dataloader))
# print("number test:", len(test_dataloader))

train_loader = CombinedLoader([labeled_dataloader, unlabeled_dataloader], mode="max_size_cycle")

trainer = L.Trainer(
    max_steps=cfg["steps"], 
    enable_checkpointing=False, 
    log_every_n_steps=100, 
    check_val_every_n_epoch=None, 
    val_check_interval=cfg["val_check_interval"],
    # accumulate_grad_batches = 1
    benchmark=True,
    )
trainer.fit(model=MeanTeacher(), train_dataloaders=train_loader, val_dataloaders=test_dataloader)
