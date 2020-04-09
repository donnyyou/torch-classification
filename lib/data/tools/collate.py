#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Adapted from https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/collate.py


import random
import collections
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch._six import string_classes, int_classes

from lib.utils.parallel.data_container import DataContainer
from lib.utils.helpers.tensor_helper import TensorHelper


def stack(batch, data_key=None, device_ids=None):
    if isinstance(batch[0][data_key], DataContainer):
        if batch[0][data_key].stack:
            assert isinstance(batch[0][data_key].data, torch.Tensor) or \
                   isinstance(batch[0][data_key].data, int_classes) or \
                   isinstance(batch[0][data_key].data, float) or \
                   isinstance(batch[0][data_key].data, string_classes) or \
                   isinstance(batch[0][data_key].data, collections.Mapping) or \
                   isinstance(batch[0][data_key].data, collections.Sequence)
            stacked = []
            if batch[0][data_key].samples_per_gpu and len(device_ids) > 1:
                samples_per_gpu = (len(batch) - 1 + len(device_ids)) // len(device_ids)
                for i in range(0, len(batch), samples_per_gpu):
                    stacked.append(
                        default_collate([sample[data_key].data for sample in batch[i:i + samples_per_gpu]])
                    )
            else:
                stacked = default_collate([sample[data_key].data for sample in batch])

            if batch[0][data_key].return_dc and len(device_ids) > 1:
                return DataContainer(stacked, stack=batch[0][data_key].stack,
                                     samples_per_gpu=batch[0][data_key].samples_per_gpu,
                                     cpu_only=batch[0][data_key].cpu_only)
            else:
                return stacked
        else:
            stacked = []
            if batch[0][data_key].samples_per_gpu and len(device_ids) > 1:
                samples_per_gpu = (len(batch) - 1 + len(device_ids)) // len(device_ids)
                for i in range(0, len(batch), samples_per_gpu):
                    stacked.append([sample[data_key].data for sample in batch[i:i + samples_per_gpu]])
            else:
                stacked = [sample[data_key].data for sample in batch]

            if batch[0][data_key].return_dc and len(device_ids) > 1:
                return DataContainer(stacked, stack=batch[0][data_key].stack,
                                     samples_per_gpu=batch[0][data_key].samples_per_gpu,
                                     cpu_only=batch[0][data_key].cpu_only)
            else:
                return stacked
    else:
        return default_collate([sample[data_key] for sample in batch])


def collate(batch, device_ids=None):
    device_ids = list(range(torch.cuda.device_count())) if device_ids is None else device_ids
    data_keys = batch[0].keys()
    return dict({key: stack(batch, data_key=key, device_ids=device_ids) for key in data_keys})
