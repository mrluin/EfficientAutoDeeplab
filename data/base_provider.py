import numpy as np
import torch
from torch.utils.data import Subset


class DataProvider:
    # random seed for the validation set
    VALID_SEED = 0

    @staticmethod
    def name():
        # name of the dataset
        raise NotImplementedError

    @property
    def data_shape(self):
        # shape as python list of one data entry
        raise NotImplementedError

    @property
    def nb_classes(self):
        raise NotImplementedError

    @property
    def save_path(self):
        raise NotImplementedError

    @staticmethod
    def random_sample_valid_set(train_dataset, valid_size):
        # random split training set into train and valid,
        # which are used to train network parameters and architecture parameters, respectively.
        # now we have four 'dataset': train, valid from train, true valid, and test
        # only perform transformation on train
        train_size = len(train_dataset)
        assert train_size > valid_size, 'size of train_set is larger than valid_set'

        indices = list(range(train_size))
        split = int(np.floor(train_size * valid_size)) # split point

        # TODO shuffle
        g = torch.Generator()
        g.manual_seed(DataProvider.VALID_SEED)
        rand_indices = torch.randperm(train_size, generator=g).tolist()

        train_indices, valid_indices = rand_indices[split:], rand_indices[:split]

        return train_indices, valid_indices


