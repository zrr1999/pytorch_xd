import torch
from model import se_resnext50
import argparse
from pytorch_xd.trainer import DatasetTrainer
from pytorch_xd.dataset import KFoldDataset


# ==============参数语句块=================#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parse = argparse.ArgumentParser(description="海洋生物细粒度分类训练")
parse.add_argument('--data_path', type=str, default='./data', help='数据集data文件的位置')
parse.add_argument('--lr', type=int, default=3e-4, help='优化器的学习率')
parse.add_argument('--batchsize', type=int, default=64, help='每一个batch的数据个数')
parse.add_argument('--max_epoch', type=int, default=2, help='epoch总数')
parse.add_argument('--crop_size', type=int, default=96, help='随机裁切的大小')
opt = parse.parse_args()
print(opt)
# ========================================#


dataset = KFoldDataset(opt.data_path, opt.crop_size, n_splits=10)
trainer = DatasetTrainer(device, opt, dataset=dataset)
for n in range(10):
    model = se_resnext50()
    trainer.train(model, opt.max_epoch, f"Model{n}")
    dataset.set_mode(n=1)
