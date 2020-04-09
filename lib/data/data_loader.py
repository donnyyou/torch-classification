#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Class for the Pose Data Loader.


from torch.utils import data

import lib.data.tools.pil_aug_transforms as pil_aug_trans
import lib.data.tools.cv2_aug_transforms as cv2_aug_trans
import lib.data.tools.transforms as trans
from lib.data.tools.collate import collate
from lib.data.tools.sampler import RankingSampler, BalanceSampler
from lib.data.datasets.default_dataset import DefaultDataset
from lib.data.datasets.test_datasets import TestDefaultDataset, TestListDataset
from lib.utils.tools.logger import Logger as Log


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_val_transform = pil_aug_trans.PILAugCompose(self.configer, split='val')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='val')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_test_transform = pil_aug_trans.PILAugCompose(self.configer, split='test')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_test_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='test')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
        ])

    def get_trainloader(self, loader_type=None, data_dir=None,
                        batch_size=None, samples_per_class=None, min_cnt=None, max_cnt=None):
        loader_type = self.configer.get('train', 'loader') if loader_type is None else loader_type
        data_dir = self.configer.get('data', 'data_dir') if data_dir is None else data_dir
        batch_size = self.configer.get('train', 'batch_size') if batch_size is None else batch_size
        min_cnt = self.configer.get('train', 'min_count') if min_cnt is None else min_cnt
        max_cnt = self.configer.get('train', 'max_count') if max_cnt is None else max_cnt
        samples_per_class = self.configer.get('train', 'samples_per_class') if samples_per_class is None else samples_per_class
        if loader_type is None or loader_type == 'default':
            dataset = DefaultDataset(data_dir=data_dir, dataset='train',
                                     aug_transform=self.aug_train_transform,
                                     img_transform=self.img_transform, configer=self.configer)
            trainloader = data.DataLoader(
                dataset=dataset,
                batch_sampler=BalanceSampler(
                    label_list=dataset.label_list, batch_size=batch_size, min_cnt=min_cnt, max_cnt=max_cnt
                ),
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return trainloader

        elif loader_type == 'ranking':
            dataset = DefaultDataset(data_dir=data_dir, dataset='train',
                                     aug_transform=self.aug_train_transform,
                                     img_transform=self.img_transform, configer=self.configer)
            trainloader = data.DataLoader(
                dataset=dataset,
                batch_sampler=RankingSampler(
                    label_list=dataset.label_list, samples_per_class=samples_per_class,
                    batch_size=batch_size, min_cnt=min_cnt, max_cnt=max_cnt
                ),
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return trainloader

        else:
            Log.error('{} train loader is invalid.'.format(self.configer.get('train', 'loader')))
            exit(1)

    def get_valloader(self, loader_type=None, data_dir=None, batch_size=None):
        loader_type = self.configer.get('val', 'loader') if loader_type is None else loader_type
        data_dir = self.configer.get('data', 'data_dir') if data_dir is None else data_dir
        batch_size = self.configer.get('val', 'batch_size') if batch_size is None else batch_size
        if loader_type is None or loader_type == 'default':
            valloader = data.DataLoader(
                DefaultDataset(data_dir=data_dir, dataset='val',
                               aug_transform=self.aug_val_transform,
                               img_transform=self.img_transform, configer=self.configer),
                batch_size=batch_size, shuffle=False,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return valloader

        else:
            Log.error('{} val loader is invalid.'.format(self.configer.get('val', 'loader')))
            exit(1)

    def get_testloader(self, test_dir=None, list_path=None, data_dir="/", batch_size=1, workers=8):
        if test_dir is not None:
            assert list_path is None
            testloader = data.DataLoader(
                TestDefaultDataset(test_dir=test_dir,
                                   aug_transform=self.aug_test_transform,
                                   img_transform=self.img_transform,
                                   configer=self.configer),
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True, collate_fn=collate
            )
            return testloader

        elif list_path is not None:
            testloader = data.DataLoader(
                TestListDataset(data_dir=data_dir,
                                list_path=list_path,
                                aug_transform=self.aug_test_transform,
                                img_transform=self.img_transform,
                                configer=self.configer),
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True, collate_fn=collate
            )
            return testloader

        else:
            Log.error('Params is invalid.')
            exit(1)


if __name__ == "__main__":
    # Test data loader.
    pass
