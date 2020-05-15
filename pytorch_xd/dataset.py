from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import KFold
import torch
import csv
import PIL.Image as Image
import numpy as np


class DatasetBase(Dataset):

    def __init__(self):
        super().__init__()
        self.n = 0
        self.mode = None

    def set_mode(self, mode, n):
        self.mode = mode
        self.n += n


class DemoDataset(Dataset):
    def __init__(self, image_pth, crop_size):
        super().__init__()
        self.image_pth = image_pth
        self.crop_size = crop_size

        self.to_transform = transforms.Compose([
            transforms.Resize(550),
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20)
        ])  # *数据增强：随机切割,水平随机翻转,垂直随机翻转,90度随机旋转
        self.to_Tensor = transforms.ToTensor()
        with open('training.csv', 'r') as f:
            reader = csv.reader(f)
            self.result = np.array(list(reader))
        self.result = self.result[:10]

    def __getitem__(self, idx):
        image_path = self.image_pth+'/'+str(self.result[idx][0])+'.jpg'
        label = self.result[idx][1]
        # label = create_one_hot(20,int(label))
        # print('label',label)
        label = int(label)
        # label = torch.LongTensor(label)
        img = Image.open(image_path)
        if self.to_transform is not None:
            img = self.to_transform(img)
        img = self.to_Tensor(img)
        return img, label

    def __len__(self):
        return len(self.result)


class KFoldDataset(DatasetBase):
    def __init__(self, image_pth, crop_size, n_splits=10):
        super().__init__()
        self.image_pth = image_pth
        self.crop_size = crop_size
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=5)

        self.to_transform = transforms.Compose([
            transforms.Resize(550),
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20)
        ])  # *数据增强：随机切割,水平随机翻转,垂直随机翻转,90度随机旋转
        self.to_Tensor = transforms.ToTensor()
        with open('training.csv', 'r') as f:
            reader = csv.reader(f)
            self.result = np.array(list(reader))
        self.results = self.result
        self.results_idx = list(kf.split(self.result))

    def set_mode(self, mode="train", n=0):
        # 预处理transform
        if mode == "train":
            self.to_transform = transforms.Compose([
                transforms.Resize(550),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20)
            ])  # *数据增强：随机切割,水平随机翻转,垂直随机翻转,90度随机旋转
        else:
            self.to_transform = transforms.Compose([
                transforms.Resize(550)
            ])
        if mode == "train":
            self.result = self.results[self.results_idx[self.n][0]]
        else:
            self.result = self.results[self.results_idx[self.n][1]]
        super().set_mode(mode, n)

    def __getitem__(self, idx):
        image_path = self.image_pth+'/'+str(self.result[idx][0])+'.jpg'
        label = self.result[idx][1]
        # label = create_one_hot(20,int(label))
        # print('label',label)
        label = int(label)
        # label = torch.LongTensor(label)
        img = Image.open(image_path)
        if self.to_transform is not None:
            img = self.to_transform(img)
        img = self.to_Tensor(img)
        return img, label

    def __len__(self):
        return len(self.result)
