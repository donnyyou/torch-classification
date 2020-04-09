#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch.nn as nn

from lib.models.nets.deploy_cls_model import DeployClsModel


class DeployDistillModel(nn.Module):
    def __init__(self, configer):
        super(DeployDistillModel, self).__init__()
        self.configer = configer
        self.main = DeployClsModel(self.configer, flag='main')
        self.peer = DeployClsModel(self.configer, flag='peer')

    def forward(self, x):
        if self.configer.get('deploy', 'net_type') == 'main':
            x = self.main(x)

        elif self.configer.get('deploy', 'net_type') == 'peer':
            x = self.peer(x)
        else:
            raise Exception('Not implemented!!!')

        return x
