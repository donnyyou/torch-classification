#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn

import lib.models.slim as slim
from lib.models.tools.module_helper import ModuleHelper
from lib.models.loss.loss import BASE_LOSS_DICT


LOSS_TYPE = {
    'ce_loss': {
        'ce_loss0': 1.0
    },
}


class BaseModel(nn.Module):
    def __init__(self, configer):
        super(BaseModel, self).__init__()
        self.configer = configer
        self.net = slim.__dict__[configer.get('network.backbone')](num_classes=configer.get('data.num_classes')[0],
                                                                   has_classifier=True)
        self.valid_loss_dict = LOSS_TYPE[configer.get('loss', 'loss_type')]

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.net.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.net.load_state_dict(state_dict, strict)

    def forward(self, data_dict):
        out_dict = dict()
        label_dict = dict()
        loss_dict = dict()
        in_img = ModuleHelper.preprocess(data_dict['img'], self.configer.get('data.normalize'))
        out = self.net(in_img)
        out_dict['out'] = out
        label_dict['out'] = data_dict['label'][:, 0]
        if 'ce_loss' in self.valid_loss_dict:
            loss_dict['ce_loss'] = dict(
                params=[out, data_dict['label'][:, 0]],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_loss']])
            )

        return out_dict, label_dict, loss_dict
