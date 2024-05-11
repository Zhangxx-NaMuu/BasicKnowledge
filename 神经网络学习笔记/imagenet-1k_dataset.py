# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> imagenet-1k_dataset
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2024/5/11 15:43
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2024/5/11 15:43:
==================================================  
"""
__author__ = 'zxx'
"""
因为ILSVRC2012_img_val文件中的图片没有按标签放到制定的文件夹中，故该代码根据ILSVRC2012_devkit_t12中的标签信息
将ILSVRC2012_img_val文件中的图片分类放到制定的文件夹中，方便使用dataloader进行加载。
数据下载地址：https://blog.csdn.net/Yuan_mingyu/article/details/123940228
验证集处理方式：（参考链接：https://zhuanlan.zhihu.com/p/370799616#:~:text=%E8%B6%85%E8%BF%871400%E4%B8%87%E7%9A%84%E5%9B%BE%E5%83%8FURL%E8%A2%ABImageNet%E6%89%8B%E5%8A%A8%E6%B3%A8%E9%87%8A%EF%BC%8C%E4%BB%A5%E6%8C%87%E7%A4%BA%E5%9B%BE%E7%89%87%E4%B8%AD%E7%9A%84%E5%AF%B9%E8%B1%A1%3B%E5%9C%A8%E8%87%B3%E5%B0%91%E4%B8%80%E7%99%BE%E4%B8%87%E5%BC%A0%E5%9B%BE%E5%83%8F%E4%B8%AD%EF%BC%8C%E8%BF%98%E6%8F%90%E4%BE%9B%E4%BA%86%E8%BE%B9%E7%95%8C%E6%A1%86%E3%80%82%20ImageNet%E5%8C%85%E5%90%AB2%E4%B8%87%E5%A4%9A%E4%B8%AA%E7%B1%BB%E5%88%AB%3B%20%E4%B8%80%E4%B8%AA%E5%85%B8%E5%9E%8B%E7%9A%84%E7%B1%BB%E5%88%AB%EF%BC%8C%E5%A6%82%E2%80%9C%E6%B0%94%E7%90%83%E2%80%9D%E6%88%96%E2%80%9C%E8%8D%89%E8%8E%93%E2%80%9D%EF%BC%8C%E6%AF%8F%E4%B8%AA%E7%B1%BB%E5%8C%85%E5%90%AB%E6%95%B0%E7%99%BE%E5%BC%A0%E5%9B%BE%E5%83%8F%E3%80%82%20%E5%9C%A8%E4%B8%80%E4%BA%9B%E8%AE%BA%E6%96%87%E4%B8%AD%EF%BC%8C%E6%9C%89%E7%9A%84%E4%BA%BA%E4%BC%9A%E5%B0%86%E8%BF%99%E4%B8%AA%E6%95%B0%E6%8D%AE%E5%8F%AB%E6%88%90ImageNet,1K%20%E6%88%96%E8%80%85ISLVRC2012%EF%BC%8C%E4%B8%A4%E8%80%85%E6%98%AF%E4%B8%80%E6%A0%B7%E7%9A%84%E3%80%82%20%E2%80%9C1%20K%E2%80%9D%E4%BB%A3%E8%A1%A8%E7%9A%84%E6%98%AF1000%E4%B8%AA%E7%B1%BB%E5%88%AB%E3%80%82）
创建val文件夹，将val.tar移动到val文件中，cd到val文件，解压
1、mkdir val--> mv ILSVRC2012_img_val.tar val-->cd val-->tar -xvf ILSVRC2012_img_val.tar
# 重新分类
2、 Invoke-WebRequest -Uri https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh -OutFile valprep.sh ##下载sh文件
--> bash valprep.sh根据.sh文件分类验证集

# 在处理数据集的时候，target标签是根据文件名的排序ID来确定的，如：n01440764，这个文件名可以在valprep.sh中获取，并将相应的图片移动到相应的文件夹中，假设一共有1000个文件夹，
n01440764在这1000个文件夹中排序为1，则他的标签为0.
"""

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# yong pytorch 加载数据
def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    # traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.RandomResizedCrop((480, 416)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((480, 416)),
            transforms.CenterCrop((480, 416)),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader


# 用tim加载数据
from timm.data import ImageDataset, create_dataset, create_loader


def data_loader1():
    dataset_eval = create_dataset(
        name='ILSVRC2012_img_val', root=r"D:\code\zxx_code\FasterViT\data", split="val", is_training=False,
        class_map='',
        download=False,
        batch_size=1)

    from PIL import Image
    print(len(dataset_eval))
    print(dataset_eval[1110])
    # plt.figure("dog")
    # plt.imshow(dataset_eval[6][0])
    # plt.show()

    loader_eval = create_loader(
        dataset_eval,
        input_size=[3, 480, 416],
        batch_size=1,
        is_training=False,
        use_prefetcher=False,
        interpolation='',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=4,
        distributed=False,
        crop_pct=0.875,
        pin_memory=False,
        fp16=True
    )
    return loader_eval
