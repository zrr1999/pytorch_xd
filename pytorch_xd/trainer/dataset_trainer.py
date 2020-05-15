import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_xd.dataset import KFoldDataset
from .loss_trainer import LossTrainer


class DatasetTrainer(LossTrainer):

    def __init__(self, device, opt, loss_fn=None, dataset=None):
        """

        Args:
            opt: 训练参数，包括学习率（lr）等。
        """
        super().__init__(device, opt, loss_fn)
        self.dataset = dataset

    def configure_optimizers(self, model):
        """
        得到用于指定模型的优化器。

        Args:
            model: 训练器需要训练的模型。

        Returns: 优化器

        """
        optimizer = optim.Adam(model.parameters(), lr=self.opt.lr, betas=(0.9, 0.999))
        # scheduler = StepLR(optimizer, step_size=1)
        return optimizer

    def train_dataloader(self):
        self.dataset.set_mode("train")
        loader = DataLoader(self.dataset, batch_size=self.opt.batchsize, shuffle=True)
        return loader

    def val_dataloader(self):
        self.dataset.set_mode("val")
        loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        return loader

    def test_dataloader(self):
        self.dataset.set_mode("test")
        loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        return loader
