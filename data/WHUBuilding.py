import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import os
import glob
import random
import torch
from PIL import Image
from data.base_provider import *
class WHUBuildingDataset(data.Dataset):
    def __init__(self, data_path, transform=True, rt_filename=False):
        super(WHUBuildingDataset, self).__init__()

        #assert subset == 'train' or subset == 'valid' or subset == 'test',\
        #    'subset can be only chose in [train, valid, test]'

        self.data_path = data_path
        self.transform = transform
        self.rt_filename = rt_filename

        self.data_list = glob.glob(os.path.join(
            self.data_path,
            'image',
            '*'
        ))
        self.target_list = glob.glob(os.path.join(
            self.data_path,
            'label',
            '*'
        ))
        self.mapping = {
            0 : 0,
            255 : 1,
        }
        self.means, self.stds = [0.4353, 0.4452, 0.4131], [0.2044, 0.1924, 0.2013]

    def mask_to_label(self, targets):

        for key in self.mapping.keys():
            targets[targets == key] = self.mapping[key]
        return targets

    def transformations(self, datas, targets):

        # train without transform
        # train with transform
        # valid without transform
        # test without transform

        if self.transform:
            if random.random() > 0.5:
                datas = TF.hflip(datas)
                targets = TF.hflip(targets)

            if random.random() > 0.5:
                datas = TF.vflip(datas)
                targets = TF.vflip(targets)

        # for datas, transformations(PIL), to_tensor, normalize
        datas = TF.to_tensor((datas))
        datas = TF.normalize(datas, mean=self.means, std=self.stds)
        # for targets, transformations(PIL), to_tensor, mask_to_label(uint8), long()
        targets = torch.from_numpy(np.array(targets, dtype=np.uint8))
        targets = self.mask_to_label(targets)
        targets = targets.long()

        return datas, targets

    def __getitem__(self, index):

        datas = Image.open(self.data_list[index])
        targets = Image.open(self.target_list[index])
        if self.rt_filename:
            return self.transformations(datas, targets), self.data_list[index]
        else:
            return self.transformations(datas, targets)

    def __len__(self):
        return len(self.data_list)

class WHUBuildingDataProvider(DataProvider):
    def __init__(self,
                 save_path,
                 train_batch_size,
                 valid_size,
                 valid_batch_size,
                 test_batch_size,
                 nb_works):
        # valid size means proportion w.r.t. training set
        super(WHUBuildingDataProvider, self).__init__()

        # dataset root directory
        self._save_path = save_path
        train_dataset = WHUBuildingDataset(self.train_path, transform=True, rt_filename=False)

        # if valid_size is not None, using valid_set split from training set
        # if valid_size is None, using original validation set
        if valid_size is not None:
            # There are four dataset, training dataset, valid from train, validation, and testing
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: {}'.format(valid_size)
            train_indices, valid_indices = self.random_sample_valid_set(train_dataset, valid_size)

            train_sampler = data.SubsetRandomSampler(train_indices)
            valid_sampler = data.SubsetRandomSampler(valid_indices)

            valid_dataset = WHUBuildingDataset(self.train_path, transform=False, rt_filename=False)

            # TODO pin_memory large cpu consumption
            self.train_loader = data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=nb_works, pin_memory=False,
            )
            # valid batch_size = test_batch_size
            self.valid_loader = data.DataLoader(
                valid_dataset, batch_size=valid_batch_size, sampler=valid_sampler,
                num_workers=nb_works, pin_memory=False,
            )

            true_valid_dataset = WHUBuildingDataset(self.valid_path, transform=False, rt_filename=False)
            self.true_valid_loader = data.DataLoader(
                true_valid_dataset, batch_size=valid_batch_size, shuffle=False,
                num_workers=nb_works, pin_memory=False,
            )
        else:
            # There are three dataset, training set, validation set, and testing set
            self.train_loader = data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=nb_works, pin_memory=False
            )
            valid_dataset = WHUBuildingDataset(self.valid_path, transform=False, rt_filename=False)
            self.valid_loader = data.DataLoader(
                valid_dataset, batch_size=valid_batch_size, shuffle=False,
                num_workers=nb_works, pin_memory=False
            )

        # test in the whole end-to-end process, do not return filename
        # when performing prediction, return filename
        test_dataset = WHUBuildingDataset(self.test_path, transform=False, rt_filename=True)
        self.test_loader = data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=nb_works, pin_memory=False,
        )
        # cannot reach this case
        if self.valid_loader is None:
            self.valid_loader = self.true_valid_loader

    @staticmethod
    def name():
        return 'WHUBuilding'

    @property
    def data_shape(self):
        # in [C H W]
        return 3, self.image_size, self.image_size

    @property
    def nb_classes(self):
        return 2

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/jingweipeng/ljb/WHUBuilding'
        return self._save_path

    @property
    def train_path(self):
        return os.path.join(self._save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'valid')

    @property
    def test_path(self):
        return os.path.join(self._save_path, 'test')

    def build_train_transform(self):
        # TODO chose transformation to perform dynamically
        raise NotImplementedError

    @property
    def origin_size(self):
        return 512

    @property
    def image_size(self):
        return 512

