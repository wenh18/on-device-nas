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
from tools.valdata_idx import get_val_idx
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

        ## resnet50
        # self.layer40downsample = []
        # self.layer41conv1 = []
        # self.layer41conv2 = []
        # self.layer41conv3 = []
        # self.layer42conv1 = []
        # self.layer42conv2 = []
        # self.layer42conv3 = []
        
        ##resnet20
        self.layer30downsample1 = []
        self.layer31conv1 = []
        self.layer31conv2 = []
        self.layer32conv1 = []
        self.layer32conv2 = []
        

        ##vgg19
        #self.vgg_label = []

        # now load the picked numpy arrays
        if self.train:
            file_path = '/root/resnet20/tmp_resnet20/expert_train'
        else:
            file_path = '/root/resnet20/tmp_resnet20/expert_test'
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

            # self.layer40downsample.extend(entry['layer40downsample'])
            # self.layer41conv1.extend(entry['layer41conv1'])
            # self.layer41conv2.extend(entry['layer41conv2'])
            # self.layer41conv3.extend(entry['layer41conv3'])
            # self.layer42conv1.extend(entry['layer42conv1'])
            # self.layer42conv2.extend(entry['layer42conv2'])
            # self.layer42conv3.extend(entry['layer42conv3'])

            self.layer30downsample1.extend(entry['layer30downsample1'])
            self.layer31conv1.extend(entry['layer31conv1'])
            self.layer31conv2.extend(entry['layer31conv2'])
            self.layer32conv1.extend(entry['layer32conv1'])
            self.layer32conv2.extend(entry['layer32conv2'])

            #self.vgg_label.extend(entry['vgg_label'])

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
        target = list((target, self.layer30downsample1[index],self.layer31conv1[index],self.layer31conv2[index],
                        self.layer32conv1[index],self.layer32conv2[index]))
        # target = list((target, self.layer40downsample[index],self.layer41conv1[index],self.layer41conv2[index],
        #             self.layer41conv3[index],self.layer42conv1[index],self.layer42conv2[index],self.layer42conv3[index]))
        #target = list((target,self.vgg_label[index]))
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
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def load_dis_data(batch_size=512,workers=14,train=True,index=0):
    normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2761])
    if train:
        CIFAR100_TRAIN = CIFAR100_DOUBLELABEL(root=r'/root/resnet20/cifar-100-python', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        train_idxs = generate_idxs(batchsize=batch_size,train=True)
        train_sampler = mysampler(CIFAR100_TRAIN)
        train_sampler.set_idxs(train_idxs)
        loader = torch.utils.data.DataLoader(
                        CIFAR100_TRAIN,
                        batch_size=batch_size, shuffle=False,
                        sampler=train_sampler,
                        num_workers=workers, pin_memory=True)
    else:
        CIFAR100_TEST = CIFAR100_DOUBLELABEL(root=r'/root/resnet20/cifar-100-python', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True) 
        #test_idxs = generate_idxs(batchsize=10000,num_class=100,train=False)  
        test_idxs = get_val_idx(index)
        test_sampler = mysampler(CIFAR100_TEST)
        test_sampler.set_idxs(test_idxs)
        loader = torch.utils.data.DataLoader(
                        CIFAR100_TEST,
                        batch_size=10000, shuffle=False,
                        sampler=test_sampler,
                        num_workers=workers, pin_memory=True)
                        
    return loader
