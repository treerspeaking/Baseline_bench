import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torchmetrics import Accuracy
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np

from data import (
    MySVHN
)
from model import MTConvNet

import itertools

class MeanTeacher(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.student = MTConvNet(10)
        self.teacher = MTConvNet(10)
        # self.teacher = self.teacher.requires_grad_(False)
        for param in self.teacher.parameters():
                param.detach_()
        self.alpha = 0.999
        self.ce_loss = torch.nn.CrossEntropyLoss(size_average=False)
        self.mse_loss = torch.nn.MSELoss(size_average=False)
        self.acc = Accuracy('multiclass', num_classes=10)
        self.consistency_weight = 10
        
    def ema(self, student, teacher, alpha):
        
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        with torch.no_grad():
            # Update parameters
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(alpha).add_((1 - alpha) * s_param.data)
            
            # # Update only floating-point buffers
            # for t_buffer, s_buffer in zip(teacher.buffers(), student.buffers()):
            #     if t_buffer.dtype.is_floating_point:
            #         t_buffer.data.mul_(alpha).add_((1 - alpha) * s_buffer.data)
            # # Non-floating-point buffers (e.g., num_batches_tracked) are left unchanged
            
            
        
    def training_step(self, batch, batch_idx):
        
        # print(batch)
        labeled_data_s, labeled_data_t, label = batch[0]
        unlabeled_data_s, unlabeled_data_t = batch[1]
        labeled_data_len = len(labeled_data_s)
        combine_data_s = torch.concat((labeled_data_s, unlabeled_data_s), dim=0)
        combine_data_t = torch.concat((labeled_data_t, unlabeled_data_t), dim=0)
        stu_out = self.student(combine_data_s)
        tea_out = self.teacher(combine_data_t)
        num_classes = stu_out.size()[1]
        mini_batch = len(combine_data_s)
        loss_l = self.ce_loss(stu_out[:labeled_data_len], label) / mini_batch * 10
        loss_u = self.mse_loss(F.softmax(stu_out, dim=1), F.softmax(tea_out, dim=1)) / ( mini_batch ) * 10 * self.consistency_weight
        
        
        self.log_dict({
            "train_loss_l": loss_l,
            "train_loss_u": loss_u,
            "learning_rate": self.trainer.optimizers[0].param_groups[0]['lr']
        })
        return loss_l + self.ramp_up(self.global_step) * loss_u
        
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # return super().on_train_batch_end(outputs, batch, batch_idx)
        # ema code
        if self.global_step >= 40000:
            self.ema(self.student, self.teacher, 0.999)
        else:
            self.ema(self.student, self.teacher, 0.99)
        
        return 
    
    def validation_step(self, batch, batch_idx):
        
        labeled_data, label = batch
        mini_batch = len(labeled_data)
        stu_out = self.student(labeled_data)
        tea_out = self.teacher(labeled_data)
        loss_ce_stu = self.ce_loss(stu_out, label) / mini_batch
        loss_ce_tea = self.ce_loss(tea_out, label) / mini_batch 
        num_classes = stu_out.size()[1]
        loss_cons = self.mse_loss(F.softmax(stu_out, dim=1), F.softmax(tea_out, dim=1)) / (num_classes * mini_batch) * self.consistency_weight
        
        stu_acc = self.acc(stu_out, label)
        tea_acc = self.acc(tea_out, label)
        
        class_predictions = torch.argmax(stu_out, dim=1)
        class_counts = torch.zeros(10, dtype=torch.int32)  # Assuming 10 classes
        for class_idx in range(10):
            count = (class_predictions == class_idx).sum().item()
            self.log(f"class_{class_idx}_count", count, on_epoch=True, on_step=True)
        
        self.log_dict({
            "val_loss_ce_stu": loss_ce_stu, 
            "val_loss_ce_tea": loss_ce_tea,
            "val_loss_cons": loss_cons,
            "val_stu_acc": stu_acc,
            "val_tea_acc": tea_acc
        }, on_epoch=True, on_step=True)
        
        
        
    
    @staticmethod
    def ramp_up(step):
        if (step >= 40000):
            return 1
        else:
            return np.exp(-5*(1 - step / 40000)**2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, betas=(0.9, 0.99), eps=1e-8)
        scheduler = LambdaLR(optimizer, self.ramp_up)
        
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

# cifar10_train = Cifar10(True, transforms=v2.Compose([
#     v2.ToTensor(),
#     v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
#     v2.RandomAffine(degrees=0, translate=(0.0626, 0.0626)),
#     v2.RandomHorizontalFlip(0.5),
#     v2.GaussianNoise(sigma=0.15)
# ]))

# Cifar10_test = Cifar10(False, transforms=v2.Compose([
#     v2.ToTensor(),
#     v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
# ]))

SVHN_train = MySVHN(".", split="train", transforms=v2.Compose([
    v2.ToTensor(),
    v2.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    v2.RandomAffine(degrees=0, translate=(0.0626, 0.0626)),
    v2.GaussianNoise(sigma=0.15)
]), download=True)

SVHN_test = MySVHN(".", split="test", transforms=v2.Compose([
    v2.ToTensor(),
    v2.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    v2.RandomAffine(degrees=0, translate=(0.0626, 0.0626)),
    v2.GaussianNoise(sigma=0.15)
]), download=True)

# labeled_dataloader, unlabeled_dataloader = cifar10_train.get_dataloader(labeled_batch_size=50, labeled_num_worker=4,shuffle = True, unlabeled=True, labeled_size=2000, unlabeled_batch_size=50, unlabeled_num_worker = 4, seed= None)
# test_dataloader = Cifar10_test.get_dataloader(200, 4, False)

labeled_dataloader, unlabeled_dataloader = SVHN_train.get_dataloader(labeled_batch_size=1, labeled_num_worker=4,shuffle = True, unlabeled=True, labeled_size=500, unlabeled_batch_size=99, unlabeled_num_worker = 4, seed= None)
test_dataloader = SVHN_test.get_dataloader(200, 4, False)

print("number labeled:", len(labeled_dataloader))
print("number unlabeld:", len(unlabeled_dataloader))

# combine_dataloader = CombinedDataloader(labeled_dataloader, unlabeled_dataloader)
combine_dataloader = CombinedLoader([labeled_dataloader, unlabeled_dataloader], mode="max_size_cycle")

trainer = L.Trainer(max_steps=150000, enable_checkpointing=True, log_every_n_steps=10, check_val_every_n_epoch=1)
trainer.fit(model=MeanTeacher(), train_dataloaders=combine_dataloader, val_dataloaders=test_dataloader)