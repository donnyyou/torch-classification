import numpy as np
from torch.utils.data.sampler import BatchSampler

from lib.utils.tools.logger import Logger as Log


def random_sample(label_indices_dict, min_cnt, max_cnt):
    sample_indices_dict = dict()
    for label in label_indices_dict.keys():
        target_indices = np.array(label_indices_dict[label])
        if len(target_indices) >= min_cnt:
            if max_cnt == -1 or len(target_indices) <= max_cnt:
                label_indices = target_indices
            else:
                label_indices = np.random.choice(target_indices, size=max_cnt, replace=False)
        else:
            random_indices = np.random.choice(target_indices, size=min_cnt-len(target_indices), replace=True)
            label_indices = np.append(target_indices, random_indices)

        sample_indices_dict[label] = label_indices.tolist()

    return sample_indices_dict


class RankingSampler(BatchSampler):
    """
      BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
      Returns batches of size n_classes * n_samples
      Making sure that one epoch will used all the data
    """
    def __init__(self, label_list, samples_per_class=2, batch_size=64, min_cnt=0, max_cnt=-1):
        self.label_list = label_list
        self.label_indices_dict = dict()
        for i, mlabel in enumerate(self.label_list):
            if mlabel[0] not in self.label_indices_dict:
                self.label_indices_dict[mlabel[0]] = [i]
            else:
                self.label_indices_dict[mlabel[0]].append(i)

        self.n_classes = batch_size // samples_per_class
        self.n_samples = samples_per_class
        self.min_cnt = min_cnt
        self.max_cnt = max_cnt
        assert batch_size % samples_per_class == 0
        sample_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        self.num_samples = 0
        for k in sample_dict:
            self.num_samples += len(sample_dict[k])

        Log.info('The number of resampled images is {}...'.format(self.num_samples))

    def __iter__(self):
        valid_dict = {label: True for label in self.label_indices_dict.keys()}
        label_samples_dict = {label: 0 for label in self.label_indices_dict.keys()}
        samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        for l in samples_indices_dict.keys():
            np.random.shuffle(samples_indices_dict[l])

        while len(valid_dict.keys()) >= self.n_classes:
            classes = np.random.choice(list(valid_dict.keys()), self.n_classes, replace=False)
            indices = []
            for c in classes:
                indices.extend(samples_indices_dict[c][label_samples_dict[c]:label_samples_dict[c]+self.n_samples])
                label_samples_dict[c] += self.n_samples
                if label_samples_dict[c] + self.n_samples > len(samples_indices_dict[c]):
                    del valid_dict[c]

            yield indices

    def __len__(self):

        return self.num_samples // (self.n_samples * self.n_classes)


class BalanceSampler(BatchSampler):
    def __init__(self, label_list, batch_size=64, min_cnt=0, max_cnt=-1):
        self.label_list = label_list
        self.label_indices_dict = dict()
        for i, mlabel in enumerate(self.label_list):
            if mlabel[0] not in self.label_indices_dict:
                self.label_indices_dict[mlabel[0]] = [i]
            else:
                self.label_indices_dict[mlabel[0]].append(i)

        self.batch_size = batch_size
        self.min_cnt = min_cnt
        self.max_cnt = max_cnt
        sample_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        self.num_samples = 0
        for k in sample_dict:
            self.num_samples += len(sample_dict[k])

        Log.info('The number of resampled images is {}...'.format(self.num_samples))

    def __iter__(self):
        samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)

        samples_indices = []
        for k in samples_indices_dict:
            samples_indices.extend(samples_indices_dict[k])

        np.random.shuffle(samples_indices)
        sample_index = 0
        assert len(samples_indices) > self.batch_size
        while sample_index + self.batch_size < len(samples_indices):
            yield samples_indices[sample_index:sample_index+self.batch_size]
            sample_index += self.batch_size

    def __len__(self):

        return self.num_samples // self.batch_size
