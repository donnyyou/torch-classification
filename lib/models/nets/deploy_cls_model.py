#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# ResNet in PyTorch.


import torch.nn as nn

import lib.models.slim as slim
from lib.models.tools.module_helper import ModuleHelper


class DeployClsModel(nn.Module):
    def __init__(self, configer, flag=""):
        super(DeployClsModel, self).__init__()
        self.configer = configer
        self.flag = flag if len(flag) == 0 else "{}_".format(flag)
        self.backbone = slim.__dict__[configer.get('network.{}backbone'.format(self.flag))](
            pretrained=configer.get('network.{}pretrained'.format(self.flag)),
            has_classifier=False
        )
        self.reduction = None
        fc_dim_out = configer.get('network.{}fc_dim'.format(self.flag), default=None)
        fc_dim = self.backbone.num_features
        if fc_dim_out is not None:
            self.reduction = nn.Conv2d(self.backbone.num_features, fc_dim_out, 1)
            fc_dim = fc_dim_out

        self.bn = nn.BatchNorm2d(fc_dim)

    def forward(self, x):
        x = ModuleHelper.preprocess(x, self.configer.get('data.normalize'))
        x = self.backbone(x)
        x = ModuleHelper.postprocess(x, method=self.configer.get('deploy', 'pool_type'))
        x = self.reduction(x) if self.reduction else x
        x = self.bn(x)
        x = x.flatten(1)
        x = ModuleHelper.normalize(x, method=self.configer.get('deploy', 'norm_type'))
        return x
