import logging
import numpy as np
import torch
from torch.utils import data

from data.data_parts import get_datapart
from data.transforms.make_transform import make_transform_composition


class Cifar10Dataloader(data.Dataset):
    """
    Dataloader to load the traffic signs.
    """

    def __init__(self, config, split):
        self.config = config
        self.split = split

        self.data, self.indices, self.data_file, self.labels = self.load_data_datafile_label()
        self.data = [np.transpose(np.reshape(data, [3, 32, 32]), [1, 2, 0]) for data in self.data]
        assert self.__len__(), f'No data were found in {self.config.dataset_path}.'

        self.transforms = make_transform_composition(self.config.transforms, self.split)

    def __getitem__(self, item):
        # load input image
        input_image = self.transforms(self.data[item])

        # use the labels if we have them (during train and val) otherwise use a [] placeholder
        if self.labels is not None:
            label = self.labels[item]
            label = torch.as_tensor(int(label), dtype=torch.long)
        else:
            label = []


        return {'data_files': self.data_file[item], 'indices': self.indices[item],
                'input_images': input_image, 'labels': label}

    def __len__(self):
        return len(self.data)

    def load_data_datafile_label(self):
        """
        Load data, data_filenames and labels according to the current split (train/val)
        according to the given data_part config.

        :return: [data, data_files, labels]: list of the data, the corresponding data_files and list of the corresponding labels
        """
        logging.info('Loading data')

        if self.split == 'val':
            if self.config.data_parts[0] == 'unlabelled':
                return get_datapart('unlabelled', self.config.dataset_path)
            else:
                return get_datapart('val', self.config.dataset_path)
        elif self.split == 'train':
            data, indices, data_files, labels = [], [], [], []
            for data_part_name in self.config.data_parts:
                data_part = get_datapart(data_part_name, self.config.dataset_path)
                data.extend(data_part[0])
                indices.extend(data_part[1])
                data_files.extend(data_part[2])
                labels.extend(data_part[3])
            return data, indices, data_files, labels
        else:
            return ValueError(f'Wrong split: {self.split}.')
