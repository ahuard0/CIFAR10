# -*- coding: utf-8 -*-
r"""
Created on Mon Sep  6 19:55:13 2021

Test Bench Function for the CIFAR10 dataloader.

CIFAR-10 Dataset Implementation, Built from Scratch (Test Bench)
    CIFAR10 dataset downloaded locally to A:\CIFAR\cifar-10-batches-py\
    from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    Dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton

This dataset is an implementation of a PyTorch Dataset, which is intended to be
used with a PyTorch DataLoader.

@author: Andrew Huard
"""

from CIFAR10 import CIFAR10
from torch.utils.data import DataLoader


dataset_val = CIFAR10(bool_train=False, bool_tensor=True)
dataloader_val = DataLoader(dataset_val)

for imgs, labels in dataloader_val:
    batch_size = imgs.shape[0]
    break
