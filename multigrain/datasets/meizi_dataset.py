# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Imagenet is either distributed along with a devkit to get the validation labels,
 or with the validation set reorganized into different subsets.
Here we support both.
Keeps an index of the images for fast initialization.
"""

import torch
from torch.utils import data
import os
from os import path as osp
import numpy as np
from .loader import loader as default_loader
from multigrain.utils import ifmakedirs

import ipdb
from .ImageFolderWithPaths import ImageFolderWithPaths


class meizi_dataset(data.Dataset):
    """
    ImageNet 1K dataset
    Classes numbered from 0 to 999 inclusive
    Can deal both with ImageNet original structure and the common "reorganized" validation dataset
    Caches list of files for faster reloading.
    """
    NUM_CLASSES = 1000
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    EIG_VALS = [0.2175, 0.0188, 0.0045]
    EIG_VECS = np.array([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203]
    ])

    def __init__(self, root, transform=None, force_reindex=False, loader=default_loader, starts='1'):
        self.root = root
        self.starts = starts
        self.transform = transform
        cachefile = 'data/NONIN1K' + '_cached-list.pth'
        self.classes, self.class_to_idx, self.imgs, self.labels, self.images_subdir = self.get_dataset(cachefile, force_reindex)
        self.loader = loader

    def get_dataset(self, cachefile=None, force_reindex=False): 
        # images_subdir = "/home/meizi/short_video_pic"
        images_subdir = self.root
        print('Indexing dataset \t', images_subdir)
        
        subfiles = os.listdir(images_subdir)

        if osp.isdir(osp.join(self.root, images_subdir, subfiles[0])):  # ImageFolder
            # self.starts is used for select subset images
            print(self.starts)
            classes = [folder for folder in subfiles if folder.startswith(self.starts)]
            classes.sort()
            print(classes)
            class_to_idx = {c: i for (i, c) in enumerate(classes)}
            imgs = []
            labels = []
            for label in classes:
                label_images = os.listdir(osp.join(self.root, images_subdir, label))
                label_images = [img for img in label_images if img.endswith('.jpg')]

                label_images.sort()
                imgs.extend([osp.join(label, i) for i in label_images])
                labels.extend([class_to_idx[label] for _ in label_images])

        returns = (classes, class_to_idx, imgs, labels, images_subdir)
        if cachefile is not None:
            ifmakedirs(osp.dirname(cachefile))
            torch.save(returns, cachefile)
            print('cached to', cachefile)

        return returns
        
    def __getitem__(self, idx):
        image = self.loader(osp.join(self.root, self.images_subdir, self.imgs[idx]))
        if self.transform is not None:
            image = self.transform(image)

        file_info = self.imgs[idx].split('/')
        vid = file_info[0]
        frame_id = (file_info[1].split('_')[-1]).split('.')[0]

        return (image, self.labels[idx],  int(vid), int(frame_id)) 
    
    def __len__(self):
        return len(self.imgs)
    
    def __repr__(self):
        return "IN1K(root='{}')".format(self.root)
