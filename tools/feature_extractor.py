import argparse
import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn
from lib.datasets.data_loader import DataLoader
from lib.models.tools.model_manager import ModelManager
from lib.utils.helpers.runner_helper import RunnerHelper
from lib.utils.helpers.file_helper import FileHelper
from lib.utils.parallel.data_parallel import DataParallelModel
from lib.utils.tools.configer import Configer
from lib.utils.tools.logger import Logger as Log


class FeatureExtractor(object):
    '''
    Standard model class.
    '''

    def __init__(self, args, device='cuda'):
        if torch.cuda.device_count() == 0:
            device = 'cpu'

        self.device = torch.device(device)
        Log.info('Resuming from {}...'.format(args.model_path))
        checkpoint_dict = torch.load(args.model_path)
        self.configer = Configer(config_dict=checkpoint_dict['config_dict'], args_parser=args, valid_flag="deploy")
        self.net = ModelManager(self.configer).get_deploy_model()
        RunnerHelper.load_state_dict(self.net, checkpoint_dict['state_dict'], False)
        if device == 'cuda':
            self.net = DataParallelModel(self.net, gather_=True)

        self.net = self.net.to(self.device).eval()
        self.test_loader = DataLoader(self.configer)

    def run(self, root_dir, batch_size):
        '''
          Apply the model.
        '''
        for i, data_dict in enumerate(self.test_loader.get_testloader(test_dir=root_dir, batch_size=batch_size)):
            with torch.no_grad():
                feat = self.net(data_dict['img'])

            norm_feat_arr = feat.cpu().numpy()
            for i in range(len(data_dict['meta'])):
                save_name = '{}.feat'.format(os.path.splitext(data_dict['meta'][i]['filename'])[0])
                save_path = os.path.join('{}_feat'.format(root_dir.rstrip('/')), save_name)
                FileHelper.make_dirs(save_path, is_file=True)
                ffeat = open(save_path, 'w')
                ffeat.write("%s" % (" ".join([str(x) for x in list(norm_feat_arr[i])]) + '\n'))
                ffeat.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str,
                        dest='model_path', help='The model path.')
    parser.add_argument('--root_dir', default=None, type=str,
                        dest='root_dir', help='The root path.')
    parser.add_argument('--pool_type', default="AVE", type=str,
                        dest='deploy.pool_type', help='The norm type.')
    parser.add_argument('--norm_type', default="L2", type=str,
                        dest='deploy.norm_type', help='The norm type.')
    parser.add_argument('--net_type', default="main", type=str,
                        dest='deploy.net_type', help='The net type.')
    parser.add_argument('--gpu_list', default=[0, 1], nargs='+', type=int,
                        dest='gpu_list', help='The gpu list used.')
    parser.add_argument('--batch_size', default=256, type=int,
                        dest='batch_size', help='The batch size.')

    args = parser.parse_args()
    if args.gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in args.gpu_list)

    feature_extractor = FeatureExtractor(args, 'cuda' if args.gpu_list is not None else 'cpu')
    feature_extractor.run(args.root_dir, args.batch_size)
