import torchvision
import os
from torch.utils.data import Subset
import numpy as np
import copy
import torch
import random
from itertools import combinations
import math
from torch.utils.data.sampler import BatchSampler
import sys
from typing import Any, Callable, Optional, Tuple
import pickle
import torchvision.datasets as datasets
from PIL import Image

sys.path.append(os.getcwd())

from tools.generate_dis import generate_idxs
import torchvision.transforms as transforms

class CIFAR100_DOUBLELABEL(datasets.CIFAR100):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(datasets.CIFAR100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        self.expert_label = []

        # now load the picked numpy arrays
        if self.train:
            file_path = '/root/resnet20/tmp/expert_train'
        else:
            file_path = '/root/resnet20/tmp/expert_test'
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

            self.expert_label.extend(entry['expert_label'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        target = list((target,self.expert_label[index]))
        return img, target


class mysampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        super(mysampler, self).__init__(data_source)
        self.data_source = data_source
    def __iter__(self):
        return iter(self.idxs)

    def set_idxs(self, idxs):
        self.idxs = idxs

    def __len__(self):
        return len(self.data_source)

def load_data(batch_size=512,num_expert=10,num_class=100,workers=14):
    normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2761])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=r'/root/resnet20/cifar-100-python', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=r'/root/resnet20/cifar-100-python', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=10000, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def load_dis_data(batch_size=512,num_class=10,workers=14):

    normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2761])
    CIFAR100_TRAIN = CIFAR100_DOUBLELABEL(root=r'/root/resnet20/cifar-100-python', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    CIFAR100_TEST = CIFAR100_DOUBLELABEL(root=r'/root/resnet20/cifar-100-python', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True) 
    train_idxs = generate_idxs(train=True,batchsize=batch_size,num_class=num_class)
    test_idxs = generate_idxs(train=False,batchsize=10000,num_class=num_class)  
    train_sampler = mysampler(CIFAR100_TRAIN)
    train_sampler.set_idxs(train_idxs)
    test_sampler = mysampler(CIFAR100_TEST)
    test_sampler.set_idxs(test_idxs)
    train_loader = torch.utils.data.DataLoader(
        CIFAR100_TRAIN,
        batch_size=batch_size, shuffle=False,
        sampler=train_sampler,
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CIFAR100_TEST,
        batch_size=10000, shuffle=False,
        sampler=test_sampler,
        num_workers=workers, pin_memory=True)
    return train_loader,val_loader
