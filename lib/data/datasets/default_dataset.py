#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Default Dataset for Image Classification.


import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing

from lib.utils.parallel.data_container import DataContainer
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log


class DefaultDataset(data.Dataset):

    def __init__(self, data_dir=None, dataset=None,
                 aug_transform=None, img_transform=None, configer=None):
        super(DefaultDataset, self).__init__()
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.img_list, self.label_list = self.__read_file(data_dir, dataset)

    def __getitem__(self, index):
        img = None
        valid = True
        while img is None:
            try:
                img = ImageHelper.read_image(self.img_list[index],
                                             tool=self.configer.get('data', 'image_tool'),
                                             mode=self.configer.get('data', 'input_mode'))
                assert isinstance(img, np.ndarray) or isinstance(img, Image.Image)
            except:
                Log.warn('Invalid image path: {}'.format(self.img_list[index]))
                img = None
                valid = False
                index = (index + 1) % len(self.img_list)

        label = torch.from_numpy(np.array(self.label_list[index]))
        if self.aug_transform is not None:
            img = self.aug_transform(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            valid=valid,
            img=DataContainer(img, stack=True),
            label=DataContainer(label, stack=True)
        )

    def __len__(self):

        return len(self.img_list)

    def __read_file(self, data_dir, dataset):
        img_list = list()
        mlabel_list = list()
        img_dict = dict()
        all_img_list = []
        with open(self.configer.get('data.{}_label_path'.format(dataset)), 'r') as file_stream:
            all_img_list += file_stream.readlines()

        if dataset == 'train' and self.configer.get('data.include_val', default=False):
            with open(self.configer.get('data.val_label_path'), 'r') as file_stream:
                all_img_list += file_stream.readlines()

        for line_cnt in range(len(all_img_list)):
            line_items = all_img_list[line_cnt].strip().split(',')
            if len(line_items) == 0:
                continue

            path = line_items[0].strip()
            if not os.path.exists(os.path.join(data_dir, path)) or not ImageHelper.is_img(path):
                Log.warn('Invalid Image Path: {}'.format(os.path.join(data_dir, path)))
                continue

            img_list.append(os.path.join(data_dir, path))
            mlabel_list.append([int(item) for item in line_items[1:]])

        assert len(img_list) > 0
        Log.info('Length of {} imgs is {}...'.format(dataset, len(img_list)))
        return img_list, mlabel_list


if __name__ == "__main__":
    # Test data loader.
    pass
