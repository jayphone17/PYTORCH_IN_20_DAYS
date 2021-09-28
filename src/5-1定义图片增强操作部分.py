import numpy as np
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from torchvision import transforms, datasets
from PIL import Image
from matplotlib import pyplot as plt

# 定义图片增强操作

transform_train = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_valid = transforms.Compose([
    transforms.ToTensor()
])

def apple(x):
    return torch.tensor([x]).float()

ds_train = datasets.ImageFolder("../data/cifar2/train", transform=transform_train,
                                target_transform=apple )
ds_valid = datasets.ImageFolder('../data/cifar2/test/', transform=transform_valid,
                                target_transform=apple )

# print(ds_train.class_to_idx)
dl_train = DataLoader(ds_train,batch_size=50,shuffle=True,num_workers=3)
dl_valid = DataLoader(ds_valid,batch_size=50,shuffle=True,num_workers=3)


if __name__ == '__main__':
    for features,labels in dl_train:
        print(features.shape)
        print(labels.shape)
        break