#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Main Scripts for computer vision.


import os
import json
import random
import argparse
import functools
import torch
import torch.backends.cudnn as cudnn

from lib.utils.tools.configer import Configer
from lib.utils.tools.logger import Logger as Log


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str,
                        dest='config_file', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default="train", type=str,
                        dest='phase', help = 'The file of the hyper parameters.')
    parser.add_argument('--dtype', default="none", type=str,
                        dest='dtype', help='The dtype of the network.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')
    parser.add_argument('--workers', default=16, type=int,
                        dest='data.workers', help='The number of workers to load data.')

    # ***********  Params for data.  **********
    parser.add_argument('--data_dir', default="/data/donny/jizhuangxiang", type=str,
                        dest='data.data_dir', help='The Directory of the data.')
    parser.add_argument('--num_classes', default=None, nargs='+', type=int,
                        dest='data.num_classes', help='The number of classes.')
    parser.add_argument('--train_label_path', default="", type=str,
                        dest='data.train_label_path', help='The Label path of the train data.')
    parser.add_argument('--val_label_path', default="", type=str,
                        dest='data.val_label_path', help='The Label path of the val data.')
    parser.add_argument('--include_val', type=str2bool, nargs='?', default=False,
                        dest='data.include_val', help='Include validation set for final training.')
    parser.add_argument('--train_batch_size', default=None, type=int,
                        dest='train.batch_size', help='The batch size of training.')
    parser.add_argument('--val_batch_size', default=None, type=int,
                        dest='val.batch_size', help='The batch size of validation.')
    parser.add_argument('--train_loader', default=None, type=str,
                        dest='train.loader', help='The train loader type.')
    parser.add_argument('--val_loader', default=None, type=str,
                        dest='val.loader', help='The aux loader type.')
    parser.add_argument('--samples_per_class', default=None, type=int,
                        dest='train.samples_per_class', help='The number of samples per-class.')
    parser.add_argument('--min_count', default=0, type=int,
                        dest='train.min_count', help='The min count of per-sku.')
    parser.add_argument('--max_count', default=-1, type=int,
                        dest='train.max_count', help='The max count of per-sku.')

    # ***********  Params for augmentations.  **********
    parser.add_argument('--shuffle_trans_seq', default=None, nargs='+', type=str,
                        dest='train.aug_trans.shuffle_trans_seq', help='The augmentations transformation sequence.')
    parser.add_argument('--trans_seq', default=None, nargs='+', type=str,
                        dest='train.aug_trans.trans_seq', help='The augmentations transformation sequence.')

    for stream in ['', 'main_', 'peer_']:
        # ***********  Params for distilling.  **********
        parser.add_argument('--{}backbone'.format(stream), default=None, type=str,
                            dest='network.{}backbone'.format(stream), help='The main base network of model.')
        parser.add_argument('--{}rm_last_stride'.format(stream), type=str2bool, nargs='?', default=False,
                            dest='network.{}rm_last_stride'.format(stream), help='Whether to set last_stride=1 instead of 2.')
        parser.add_argument('--{}pretrained'.format(stream), type=str, default=None,
                            dest='network.{}pretrained'.format(stream), help='The path to peer pretrained model.')
        for branch in ['',]:
            parser.add_argument('--{}{}fc_dim'.format(stream, branch), default=None, type=int,
                                dest='network.{}{}fc_dim'.format(stream, branch), help='The dim of aux_fc features.')
            parser.add_argument('--{}{}feat_dim'.format(stream, branch), default=256, type=int,
                                dest='network.{}{}feat_dim'.format(stream, branch), help='The dim of embedding features.')
            parser.add_argument('--{}{}embed'.format(stream, branch), type=str2bool, nargs='?', default=True,
                                dest='network.{}{}embed'.format(stream, branch), help='Whether to embed features.')
            parser.add_argument('--{}{}linear_type'.format(stream, branch), default='default', type=str,
                                dest='network.{}{}linear_type'.format(stream, branch), help='The linear type of the network.')

    # ***********  Params for model.  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network.model_name', help='The name of model.')
    parser.add_argument('--checkpoints_dir', default=None, type=str,
                        dest='network.checkpoints_dir', help='The root dir of model save path.')
    parser.add_argument('--checkpoints_name', default=None, type=str,
                        dest='network.checkpoints_name', help='The name of checkpoint model.')
    parser.add_argument('--norm_type', default=None, type=str,
                        dest='network.norm_type', help='The BN type of the network.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network.resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network.resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network.resume_continue', help='Whether to continue training.')
    parser.add_argument('--resume_val', type=str2bool, nargs='?', default=False,
                        dest='network.resume_val', help='Whether to validate during resume.')
    parser.add_argument('--gather', type=str2bool, nargs='?', default=True,
                        dest='network.gather', help='Whether to gather the output of model.')
    parser.add_argument('--bb_lr_scale', default=1.0, type=float,
                        dest='network.bb_lr_scale', help='The backbone LR scale.')
    parser.add_argument('--clip_grad', type=str2bool, nargs='?', default=False,
                        dest='network.clip_grad', help='Whether to clip grad?')
    parser.add_argument('--distill_method', default=None, type=str,
                        dest='network.distill_method', help='The distill method.')

    # ***********  Params for solver.  **********
    parser.add_argument('--solver', default="solver", type=str,
                        dest='train.solver', help='The train loader type.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='solver.lr.base_lr', help='The learning rate.')
    parser.add_argument('--is_warm', type=str2bool, nargs='?', default=False,
                        dest='solver.lr.is_warm', help='Whether to warm-up for training.')
    parser.add_argument('--warm_iters', default=None, type=int,
                        dest='solver.lr.warm.warm_iters', help='The warm-up iters of training.')
    parser.add_argument('--warm_freeze', type=str2bool, nargs='?', default=False,
                        dest='solver.lr.warm.freeze', help='Whether to freeze backbone when is_warm=True')
    parser.add_argument('--max_iters', default=None, type=int,
                        dest='solver.max_iters', help='The max iters of training.')
    parser.add_argument('--display_iter', default=None, type=int,
                        dest='solver.display_iter', help='The display iteration of train logs.')
    parser.add_argument('--test_interval', default=None, type=int,
                        dest='solver.test_interval', help='The test interval of validation.')
    parser.add_argument('--save_iters', default=None, type=int,
                        dest='solver.save_iters', help='The saving iters of checkpoint model.')

    # ***********  Params for Optim Method.  **********
    parser.add_argument('--optim_method', default=None, type=str,
                        dest='solver.optim.optim_method', help='The optim method that used.')
    parser.add_argument('--sgd_wd', default=None, type=float,
                        dest='solver.optim.sgd.weight_decay', help='The weight decay for SGD.')
    parser.add_argument('--nesterov', type=str2bool, nargs='?', default=False,
                        dest='solver.optim.sgd.nesterov', help='The weight decay for SGD.')
    parser.add_argument('--adam_wd', default=None, type=float,
                        dest='solver.optim.adam.weight_decay', help='The weight decay for Adam.')

    # ***********  Params for LR Policy.  **********
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='solver.lr.lr_policy', help='The policy of lr during training.')
    parser.add_argument('--step_value', default=None, nargs='+', type=int,
                        dest='solver.lr.multistep.step_value', help='The step values for multistep.')
    parser.add_argument('--gamma', default=None, type=float,
                        dest='solver.lr.multistep.gamma', help='The gamma for multistep.')
    parser.add_argument('--power', default=None, type=float,
                        dest='solver.lr.lambda_poly.power', help='The power for lambda poly.')
    parser.add_argument('--max_power', default=None, type=float,
                        dest='solver.lr.lambda_range.max_power', help='The power for lambda range.')

    # ***********  Params for loss.  **********
    parser.add_argument('--loss_type', default=None, type=str,
                        dest='loss.loss_type', help='The loss type of the network.')

    # ***********  Params for env.  **********
    parser.add_argument('--seed', default=None, type=int, help='manual seed')
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist', type=str2bool, nargs='?', default=False,
                        dest='distributed', help='Use CUDNN.')

    args_parser = parser.parse_args()

    if args_parser.seed is not None:
        random.seed(args_parser.seed + args_parser.local_rank)
        torch.manual_seed(args_parser.seed + args_parser.local_rank)
        if args_parser.gpu is not None:
            torch.cuda.manual_seed_all(args_parser.seed + args_parser.local_rank)

    cudnn.enabled = True
    cudnn.benchmark = args_parser.cudnn
    cudnn.deterministic = True

    configer = Configer(args_parser=args_parser)
    if configer.get('gpu') is not None and not configer.get('distributed', default=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in configer.get('gpu'))

    if configer.get('network', 'norm_type') is None:
        configer.update('network.norm_type', 'batchnorm')

    if torch.cuda.device_count() <= 1 or configer.get('distributed', default=False):
        configer.update('network.gather', True)

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add('project_dir', project_dir)

    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'),
             dist_rank=configer.get('local_rank'))

    if configer.get('phase') == 'test':
        from tools.data_generator import DataGenerator
        toyset_dir = configer.get('toyset_dir', default='/workdir/donnyyou/toyset')
        DataGenerator.gen_toyset(toyset_dir)
        configer.update('data.train_label_path', os.path.join(toyset_dir, 'train_label.txt'))
        configer.update('data.val_label_path', os.path.join(toyset_dir, 'val_label.txt'))

    configer.update('logging.logfile_level', None)

    Log.info('BN Type is {}.'.format(configer.get('network', 'norm_type')))
    Log.info('Config Dict: {}'.format(json.dumps(configer.to_dict(), indent=2)))
    if configer.get('method') == 'auto_sku_merger':
        from lib.auto_sku_merger import AutoSKUMerger
        auto_sku_merger = AutoSKUMerger(configer)
        auto_sku_merger.run()
    elif configer.get('method') == 'image_classifier':
        from lib.image_classifier import ImageClassifier
        model_distiller = ImageClassifier(configer)
        model_distiller.run()
    elif configer.get('method') == 'multitask_classifier':
        from lib.multitask_classifier import MultiTaskClassifier
        multitask_distiller = MultiTaskClassifier(configer)
        multitask_distiller.run()
    else:
        Log.error('Invalid method: {}'.format(configer.get('method')))
        exit()
