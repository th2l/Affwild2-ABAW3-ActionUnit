"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import math
import pathlib
import sys

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.nn.functional import one_hot

from core.config import cfg
from core.imbalanced_sampler import ImbalancedDatasetSampler
from core.mixup import MixupTransform
from PIL import Image
from torchvision import transforms


class BGR_RGB_converts(object):
    """ Convert BGR order to RGB order and vice versa """

    def __init__(self, scale=1.):
        self.scale = scale

    def __call__(self, img):
        return self.scale * img[[2, 1, 0], :, :]


class ExpWDataset(data.Dataset):
    def __init__(self, transforms=None):
        self.root_dir = cfg.DATA_LOADER.DATA_DIR
        self.task = cfg.TASK
        data_dict = np.load(pathlib.Path(self.root_dir, 'expw_data.npy').__str__(), allow_pickle=True)
        self.data_dict = data_dict
        self.transforms = transforms

    def __len__(self):
        return len(self.data_dict)

    def get_labels(self):
        if self.task == 'EXPR':
            list_of_label = [x[:, 1].astype(np.int) for x in self.data_seqs]
            list_of_label = np.concatenate(list_of_label).flatten()
            return list_of_label

        else:
            raise ValueError('get_label() method was not implemented for {} with task {}.'.format(self.task, self.task))
        pass

    def __getitem__(self, index):
        cur_sample = self.data_dict[index]
        sample = {'image': [], 'index': cur_sample[-1].astype(np.int32).reshape(-1, 1),
                  'video_id': cur_sample[0].split('/')[1]}

        cur_img = Image.open(pathlib.Path(self.root_dir, 'ExpW', cur_sample[0])).convert('RGB')

        if self.transforms is not None:
            cur_img = self.transforms(cur_img)
        sample['image'] = [cur_img]
        sample['EXPR'] = [int(cur_sample[1])]

        sample['image'] = torch.stack(sample['image'], dim=0)
        for ky in ['VA', 'EXPR', 'AU']:
            if ky in sample.keys():
                sample[ky] = np.array(sample[ky])
                sample[ky] = torch.from_numpy(sample[ky])

        return sample


class ABAW3Dataset(data.Dataset):
    def __init__(self, split='Train', transforms=None):
        self.root_dir = cfg.DATA_LOADER.DATA_DIR
        self.task = cfg.TASK
        self.split = split
        self.seq_len = cfg.DATA_LOADER.SEQ_LEN
        self.sampling_method = cfg.DATA_LOADER.SAMPLING_METHOD
        self.transforms = transforms

        if split != 'Test':
            data_dict = \
                np.load(pathlib.Path(self.root_dir, '{}.npy'.format(self.task)).__str__(), allow_pickle=True).item()[
                    self.split]
        else:
            data_dict = np.load(pathlib.Path(self.root_dir, '{}_test.npy'.format(self.task)).__str__(),
                                allow_pickle=True).item()

        self.data_seqs = []

        for vid in data_dict:
            cur_vid_df = data_dict[vid]
            num_frames = cur_vid_df.shape[0]
            num_seqs = math.ceil(num_frames / self.seq_len)

            array_indexes = np.arange(num_frames)
            cur_set = set(array_indexes.flatten())

            # Count the number of sequences
            for idx in range(num_seqs):
                if self.sampling_method == 'sequentially' or self.seq_len == 1:
                    st_idx = idx * self.seq_len
                    ed_idx = min((idx + 1) * self.seq_len, num_frames)
                    # Get the sequence
                    cur_seq = cur_vid_df[st_idx: ed_idx, :]
                    # if self.task == 'EXPR' and cur_seq[0][1] in ['0', '7'] and torch.rand(1) < 0.5 and self.split == 'Train':
                    #     continue
                elif self.sampling_method == 'randomly':
                    if len(cur_set) > self.seq_len:
                        cur_idx = np.random.choice(np.array(list(cur_set)), self.seq_len, replace=False)
                    else:
                        cur_idx = np.array(list(cur_set))

                    cur_idx.sort()

                    cur_seq = cur_vid_df[cur_idx, :]
                    cur_set = cur_set.difference(set(cur_idx.flatten()))
                else:
                    raise ValueError('Only support sequentially or random at this time.')

                # Padding if need
                if cur_seq.shape[0] < self.seq_len:
                    # Do Padding, Pad to the end of the sequence by copying
                    cur_seq = np.pad(cur_seq, ((0, self.seq_len - cur_seq.shape[0]), (0, 0)), 'edge')

                self.data_seqs.append(cur_seq)

    def __len__(self):
        return len(self.data_seqs)

    def get_labels(self):
        if self.task == 'EXPR':
            list_of_label = [x[:, 1].astype(np.int) for x in self.data_seqs]
            list_of_label = np.concatenate(list_of_label).flatten()
            return list_of_label

        elif self.task == 'VA':
            list_of_label = [x[:, 2].astype(float) for x in self.data_seqs]  # Calculate base on valence
            list_of_label = np.concatenate(list_of_label).flatten()
            bins = [-1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.1]
            list_of_label = np.digitize(list_of_label, bins, right=False) - 1
            return list_of_label
        else:
            raise ValueError('get_label() method was not implemented for {}.'.format(self.task))
        pass

    def __getitem__(self, index):
        """
        Task VA: file_name, Valence, Arousal, Frame Index (Total 4 columns)
        Task EXPR: file_name, Emotion index (0,1,...,7), Frame index (Total 3 columns)
        Task AU: file_name, 12 action unit index (0, 1), Frame index (multi-label classification (Total 14 columns))
        Task MTL, file_name, Valence, Arousal, Emotion Index, 12 action unit index (0, 1), Video_ID, Frame index (Total 18 columns)

        :param index:
        :return:
        """

        cur_seq_sample = self.data_seqs[index]

        sample = {'image': [], 'index': cur_seq_sample[:, -1].astype(np.int32),
                  'video_id': cur_seq_sample[0, 0].split('/')[0]}
        if self.task == 'mtl':
            sample.update({'VA': [], 'EXPR': [], 'AU': []})
        else:
            sample.update({self.task: []})

        for idx in range(cur_seq_sample.shape[0]):
            # Load image
            cur_img = Image.open(pathlib.Path(self.root_dir, 'cropped_aligned', cur_seq_sample[idx, 0])).convert('RGB')

            if self.transforms is not None:
                cur_img = self.transforms(cur_img)

            sample['image'].append(cur_img)

            if self.task == 'VA':
                sample['VA'].append([float(x) for x in cur_seq_sample[idx, 1:3]])
            elif self.task in ['EXPR', 'AU']:
                sample[self.task].append([int(x) for x in cur_seq_sample[idx, 1:-1]])
            else:
                sample['VA'].append([float(x) for x in cur_seq_sample[idx, 1:3]])
                sample['EXPR'].append([int(x) for x in cur_seq_sample[idx, 3]])
                sample['AU'].append([int(x) for x in cur_seq_sample[idx, 4:-2]])

        sample['image'] = torch.stack(sample['image'], dim=0)
        for ky in ['VA', 'EXPR', 'AU']:
            if ky in sample.keys():
                sample[ky] = np.array(sample[ky])
                sample[ky] = torch.from_numpy(sample[ky])

        return sample


class ABAW3DataModule(LightningDataModule):
    def __init__(self):
        super(ABAW3DataModule, self).__init__()

        # trivial_aug = transforms.TrivialAugmentWide(num_magnitude_bins=31)
        # rand_aug = transforms.RandAugment(num_ops=9, magnitude=9, num_magnitude_bins=31)

        aug = [transforms.RandomHorizontalFlip(p=0.5),
               transforms.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25)]

        if cfg.MODEL.BACKBONE in ['vggface2-resnet50', 'vggface2-senet50']:
            # Do all transform in RGB order, then convert to BGR order
            normalize = transforms.Normalize(mean=np.array([131.0912, 103.8827, 91.4953]) / 255.0,
                                             std=[1., 1., 1.])
            train_augs = [transforms.ToTensor(), normalize, BGR_RGB_converts(scale=255.)]
            val_augs = [transforms.ToTensor(), normalize, BGR_RGB_converts(scale=255.)]
        elif cfg.MODEL.BACKBONE.split('.')[0] == 'facex':
            # Do all transform in RGB order, then convert to BGR order
            normalize = transforms.Normalize(mean=np.array([127.5, 127.5, 127.5]),
                                             std=[128., 128., 128.])
            train_augs = [transforms.ToTensor(), normalize, BGR_RGB_converts(scale=1.)]  #
            val_augs = [transforms.ToTensor(), normalize, BGR_RGB_converts(scale=1.)]  # BGR_RGB_converts(scale=1.)
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_augs = [transforms.ToTensor(), normalize]
            val_augs = [transforms.ToTensor(), normalize]

        if cfg.DATA_LOADER.IMG_SIZE < 112:
            train_augs = [transforms.RandomCrop(cfg.DATA_LOADER.IMG_SIZE), ] + train_augs
            val_augs = [transforms.CenterCrop(cfg.DATA_LOADER.IMG_SIZE), ] + val_augs

        self.transforms_train = transforms.Compose(aug + train_augs)
        self.transforms_test = transforms.Compose(val_augs)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Create train set and val set
            self.train_dataset = ABAW3Dataset(split='Train', transforms=self.transforms_train)
            self.val_dataset = ABAW3Dataset(split='Validation', transforms=self.transforms_test)

        if stage == 'test' or stage is None:
            # Create test set
            self.test_dataset = ABAW3Dataset(split='Test', transforms=self.transforms_test)

    def train_dataloader(self, shufflex=None):
        if cfg.TASK in ['EXPR', ] and cfg.DATA_LOADER.SEQ_LEN == 1:  # 'VA'
            sampler = ImbalancedDatasetSampler(self.train_dataset, replace=False, seq_len=cfg.DATA_LOADER.SEQ_LEN)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        if shufflex is not None:
            shuffle = shufflex
        return data.DataLoader(self.train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                               num_workers=cfg.DATA_LOADER.NUM_WORKERS, shuffle=shuffle, sampler=sampler)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=cfg.TEST.BATCH_SIZE,
                               num_workers=cfg.DATA_LOADER.NUM_WORKERS, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=cfg.TEST.BATCH_SIZE,
                               num_workers=cfg.DATA_LOADER.NUM_WORKERS, shuffle=False)
