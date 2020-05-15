import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader
from pytorch_xd.dataset import DemoDataset
from .loss_trainer import LossTrainer


class DemoTrainer(LossTrainer):

    def __init__(self, device, opt):
        """

        Args:
            device: 训练使用的硬件设备
            opt: 训练参数，包括学习率（lr）等
        """
        super().__init__(device, opt, nn.CrossEntropyLoss())

    def train_dataloader(self):
        dataset = DemoDataset(self.opt.data_path, self.opt.crop_size)
        loader = DataLoader(dataset, batch_size=self.opt.batchsize, shuffle=True)
        return loader

    def val_dataloader(self):
        dataset = DemoDataset(self.opt.data_path, self.opt.crop_size)
        loader = DataLoader(dataset, batch_size=self.opt.batchsize, shuffle=True)
        return loader

    def test_dataloader(self):
        dataset = DemoDataset(self.opt.data_path, self.opt.crop_size)
        loader = DataLoader(dataset, batch_size=self.opt.batchsize, shuffle=True)
        return loader
