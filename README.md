# PyTorch_xd

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)![Upload Python Package](https://github.com/zrr1999/PyTex/workflows/Upload%20Python%20Package/badge.svg)（假的）

PyTorch_xd（名称待定） 是一个高等级的PyTorch训练辅助库。

## 背景

...

## 优势
- 原生PyTorch
    1. 省事。
- 其他PyTorch训练辅助库
    1. 代码短，人类可以轻松看懂。


## 安装[![Downloads](https://pepy.tech/badge/bone-pytex)](https://pepy.tech/project/bone-pytex)（这个链接是假的）

这个项目使用 [Python](https://www.python.org/downloads/) 开发，请确保你本地安装了它。

建议使用pip安装本库。(我还没写setup，这里用不了)

```sh
$ pip install .
```

## 使用说明

使用时，你需要定义一个训练器类，其必须继承自TrainerBase同时完成所有抽象方法的编写。

```python
from pytorch_xd import TrainerBase


class TrainerDemo(TrainerBase):

    def __init__(self, opt):
        super().__init__(opt.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = opt

    def configure_optimizers(self, model):
        optimizer = optim.Adam(model.parameters(), lr=self.opt.lr, betas=(0.9, 0.999))
        # scheduler = StepLR(optimizer, step_size=1)  # 还没实现呢
        return optimizer

    def train_dataloader(self):
        dataset = TrainDataset(self.opt.data_path, self.opt.crop_size)
        loader = DataLoader(dataset, batch_size=self.opt.batchsize, shuffle=True)
        return loader

    def val_dataloader(self):
        dataset = TrainDataset(self.opt.data_path, self.opt.crop_size)
        loader = DataLoader(dataset, batch_size=self.opt.batchsize, shuffle=True)
        return loader

    def test_dataloader(self):
        dataset = TrainDataset(self.opt.data_path, self.opt.crop_size)
        loader = DataLoader(dataset, batch_size=self.opt.batchsize, shuffle=True)
        return loader

    def training_step(self, batch, batch_idx):
        inp, label = batch
        inp = inp.to(self.device)
        label = label.to(self.device)

        output_label = self.model(inp)
        loss = self.loss_fn(output_label, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        inp, label = batch
        inp = inp.to(self.device)
        label = label.to(self.device)

        output_label = self.model(inp)
        loss = self.loss_fn(output_label, label).detach()

        accuracy = torch.equal(output_label.argmax(1), label) * 1.0

        return {"loss": loss, "accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        pass
```

其实，你也可以直接导入内置的训练器类

```python
from pytorch_xd import TrainerDemo
```

下面是各种训练器类的介绍

1. **TrainerBase** 这是训练器的基础，所有其他训练器都需要继承于它，它本身是个抽象类无法直接使用。
2. **LossTrainer** 这是一个抽象训练器类，它与损失函数绑定，可以训练使用特定损失函数的一类模型。
3. **DemoTrainer** 这是一个继承自 **LossTrainer** 的训练器类，它会截取特定训练集的前十个样本训练和验证，仅用作示例，没有实际用途。
4. **DatasetTrainer** 这是一个继承自 **LossTrainer** 的训练器类，它与训练集绑定，它在实例化时需要指定一个具备自分割能力的数据集类，配合本库内置的各种数据集类可以实现交叉验证等很多功能。
5. 。。。

### API文档

我还没写

### 示例

还没写。

## 计划实现功能
1. 内置tensorboard。

## 更新日志
- (2020.05.15) v0.1.0 更新
    - 第一版。

## 维护者

[@詹荣瑞](https://github.com/tczrr1999)

## 如何贡献

非常欢迎你的加入！[提一个 Issue](https://github.com/tczrr1999/pytex/issues/new) （假的）或者提交一个 Pull Request。

### 贡献者

感谢以下参与项目的人：

## 使用许可

禁止将本辅助库及使用本辅助库制作的文档上传到百度网盘。
[GNU](LICENSE) © Rongrui Zhan
