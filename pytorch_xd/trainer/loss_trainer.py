from abc import ABC

import torch
import torch.nn as nn
import torch.optim as optim
from .trainer_base import TrainerBase


class LossTrainer(TrainerBase, ABC):
    """
    这是一个使用固定损失函数的抽象训练器。

    """

    def __init__(self, device, opt, loss_fn=None):
        """

        Args:
            device: 训练使用的硬件设备
            opt: 训练参数，包括学习率（lr）等
            loss_fn: 训练使用的损失函数
        """
        super().__init__(device)
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.opt = opt

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

    def training_step(self, batch, batch_idx):
        inp, label = batch

        output_label = self.model(inp)
        loss = self.loss_fn(output_label, label)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inp, label = batch

        output_label = self.model(inp)
        loss = self.loss_fn(output_label, label).detach()

        accuracy = torch.eq(output_label.argmax(1), label)*1.0

        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        inp, label = batch

        output_label = self.model(inp)
        loss = self.loss_fn(output_label, label).detach()

        accuracy = torch.equal(output_label.argmax(1), label) * 1.0

        return {"test_loss": loss, "test_accuracy": torch.tensor(accuracy).to(self.device)}
