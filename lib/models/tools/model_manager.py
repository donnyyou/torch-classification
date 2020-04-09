#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Cls Model for pose detection.


from lib.models.nets.cls_model import ClsModel
from lib.models.nets.distill_model import DistillModel
from lib.models.nets.deploy_distill_model import DeployDistillModel
from lib.models.nets.deploy_cls_model import DeployClsModel
from lib.models.loss.loss import Loss
from lib.utils.tools.logger import Logger as Log


CLS_MODEL_DICT = {
    'cls_model': ClsModel,
    'distill_model': DistillModel,
}

DEPLOY_MODEL_DICT = {
    'cls_model': DeployClsModel,
    'distill_model': DeployDistillModel,
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def get_model(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in CLS_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = CLS_MODEL_DICT[model_name](self.configer)

        return model

    def get_deploy_model(self, model_type=None):
        model_name = self.configer.get('network', 'model_name') if model_type is None else model_type

        if model_name not in DEPLOY_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = DEPLOY_MODEL_DICT[model_name](self.configer)

        return model

    def get_loss(self):
        if self.configer.get('network', 'gather'):
            return Loss(self.configer)

        from lib.utils.parallel.data_parallel import DataParallelCriterion
        return DataParallelCriterion(Loss(self.configer))
