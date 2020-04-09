#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


import functools
import glob
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.metric_linear import Linear, ArcLinear, CosineLinear, SphereLinear
from lib.utils.tools.logger import Logger as Log

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


class ModuleHelper(object):

    @staticmethod
    def concat(data_dict):
        out = []
        if 'img' in data_dict:
            out.append(data_dict['img'])

        if 'aux_img' in data_dict:
            out.append(data_dict['aux_img'])

        assert len(out) > 0
        return torch.cat(out, 0)

    @staticmethod
    def preprocess(x, norm_dict, mixup=0.0):
        if mixup > 0.0:
            beta = random.random() * mixup
            index = torch.randperm(x.size(0)).to(x.device)
            x = (1 - beta) * x + x[index, ...]

        x = x.div(norm_dict['div_value'])
        x = x - torch.cuda.FloatTensor(norm_dict['mean']).view(1, 3, 1, 1)
        x = x.div(torch.cuda.FloatTensor(norm_dict['std']).view(1, 3, 1, 1))
        return x

    @staticmethod
    def postprocess(feat, method='AVE', bn=None):
        if method == 'RMAC':
            feat = F.max_pool2d(feat, kernel_size=(7, 7), stride=2, padding=3)
            feat = F.adaptive_avg_pool2d(feat, 1)

        elif method == 'AVE':
            feat = F.adaptive_avg_pool2d(feat, 1)

        elif method == 'MAX':
            feat = F.adaptive_max_pool2d(feat, 1)

        elif method == 'NONE':
            pass

        else:
            raise Exception('Not implemented Error!!!')

        feat = bn(feat) if bn else feat
        return feat

    @staticmethod
    def normalize(feat, method='L2'):
        if method == 'L1':
            feat = feat / torch.sum(torch.abs(feat), dim=1, keepdim=True)
        elif method == 'L2':
            feat = feat / torch.sqrt(torch.sum(feat**2, dim=1, keepdim=True))
        elif method == 'POWER':
            ppp = 0.3
            feat = torch.sign(feat) * (torch.abs(feat) ** ppp)
        elif method == 'NONE':
            return feat
        else:
            Log.error('Norm Type {} is invalid.'.format(type))
            exit(1)

        return feat

    @staticmethod
    def Linear(linear_type):
        if linear_type == 'default':
            return Linear

        if linear_type == 'nobias':
            return functools.partial(Linear, bias=False)

        elif linear_type == 'arc0.5_30':
            return functools.partial(ArcLinear, s=30, m=0.5, easy_margin=False)

        elif linear_type == 'arc0.5_64':
            return functools.partial(ArcLinear, s=64, m=0.5, easy_margin=False)

        elif linear_type == 'easyarc0.5_30':
            return functools.partial(ArcLinear, s=30, m=0.5, easy_margin=True)

        elif linear_type == 'easyarc0.5_64':
            return functools.partial(ArcLinear, s=64, m=0.5, easy_margin=True)

        elif linear_type == 'cos0.4_30':
            return functools.partial(CosineLinear, s=30, m=0.5)

        elif linear_type == 'cos0.4_64':
            return functools.partial(CosineLinear, s=64, m=0.5)

        elif linear_type == 'sphere4':
            return functools.partial(SphereLinear, m=4)

        else:
            Log.error('Not support linear type: {}.'.format(linear_type))
            exit(1)

    @staticmethod
    def BNReLU(num_features, norm_type=None, **kwargs):
        if norm_type == 'batchnorm':
            return nn.Sequential(
                nn.BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif norm_type == 'sync_batchnorm':
            from lib.extensions.ops.sync_bn import BatchNorm2d
            return nn.Sequential(
                BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif norm_type == 'instancenorm':
            return nn.Sequential(
                nn.InstanceNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        else:
            Log.error('Not support BN type: {}.'.format(norm_type))
            exit(1)

    @staticmethod
    def BatchNorm2d(norm_type=None):
        if norm_type == 'batchnorm':
            return nn.BatchNorm2d

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm2d
        # elif bn_type == 'inplace_abn':
        #    from extensions.ops.inplace_abn.bn import InPlaceABNSync
        #    if ret_cls:
        #        return InPlaceABNSync

        #    return functools.partial(InPlaceABNSync, activation='none')

        else:
            Log.error('Not support BN type: {}.'.format(norm_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True):
        if pretrained is None:
            return model

        if not os.path.exists(pretrained):
            Log.info('{} not exists.'.format(pretrained))
            return model

        if all_match:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location="cpu")
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'prefix.{}'.format(k) in model_dict:
                    load_dict['prefix.{}'.format(k)] = v
                else:
                    load_dict[k] = v

            # load_dict = {k: v for k, v in pretrained_dict.items() if 'resinit.{}'.format(k) not in model_dict}
            model.load_state_dict(load_dict)

        else:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = model.state_dict()
            load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            Log.info('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        Log.info('Loading pretrained model:{}'.format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
