from utils import *

import torch
import torchvision
from torchvision import datasets, transforms
import pdb


trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transforms.ToTensor())

mean, std = get_mean_and_std(trainset)
print('mean, std of trainset:', mean, std)

