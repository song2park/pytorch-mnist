import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    '''
    Kaggle의 dataset을 쓰기 위해 custom dataset class를 정의하다.
    '''
    def __init__(self, filename, istrainset=True, transform=None):
        self.mnist = pd.read_csv(filename)
        self.istrainset = istrainset
        self.transform = transform

        # idx = 1 if self.istrainset else 0
        # self.mean = self.mnist.iloc[:, idx:].stack().mean()
        # self.std = self.mnist.iloc[:, idx:].values.std(ddof=1)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start_idx = 1 if self.istrainset else 0

        image = self.mnist.iloc[idx, start_idx:]
        image = np.array([image])
        # image = torch.tensor(image, dtype=torch.float32)
        image = image.astype(dtype='float32')
        if self.transform:
            image = self.transform(image)
        # print(image.dtype, label.dtype)
        image = torch.reshape(image, (-1, 28, 28))

        # testset인 경우 label이 없어 이미지만 리턴한다.
        if not self.istrainset:
            return image

        label = torch.tensor(self.mnist.iloc[idx, 0], dtype=torch.int64)

        return image, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # 1 phase : in(28,28,1)
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2 phase : in(14,14,4)
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            # 3 phase : in(7,7,4)
            nn.Linear(4 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)   # flatten tensor
        x = self.linear_layers(x)
        return x
