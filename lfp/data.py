import os
import json
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import glob
import time
from torch.nn.utils.rnn import pad_sequence

dimensions = {'Pybullet': {'obs': 18,
                           'obs_extra_info': 18,
                           'acts': 7,
                           'achieved_goals': 11,
                           'achieved_goals_extra_info': 11,
                           'shoulder_img_hw': 200,
                           'hz': 25}}


def pad_collate(batch):
    imgs = []
    obs = []
    acts = []
    seq_l = []

    for b in batch:
        imgs.append(b['imgs'])
        obs.append(b['obs'])
        acts.append(b['acts'])
        seq_l.append(b['seq_l'])

    imgs = pad_sequence(imgs, batch_first=True, padding_value=0)
    obs = pad_sequence(obs, batch_first=True, padding_value=0)
    acts = pad_sequence(acts, batch_first=True, padding_value=0)
    seq_l = torch.tensor(seq_l)

    ret = {'imgs': imgs, 'obs': obs, 'acts': acts, 'seq_l': seq_l}
    return ret


class LFP_Data(Dataset):
    def __init__(self, root, config, refresh=True):
        self.min_window_size = config.min_window_size
        self.max_window_size = config.max_window_size
        self.imgs = []
        self.trajs = []
        for sub_root in tqdm(root, file=sys.stdout):
            preload_file = os.path.join(sub_root, 'preload_' + 'window_size_' + str(config.max_window_size) + '.npy')
            if not os.path.exists(preload_file) or refresh:
                preload_imgs = []
                preload_traj = []
                traj_dirs = os.listdir(sub_root)
                for traj in traj_dirs:
                    img_seq = []

                    traj_dir = os.path.join(sub_root, traj)
                    img_dir = os.path.join(traj_dir, 'imgs')
                    img_paths = sorted([name for name in glob.glob(img_dir + '/*.png')])
                    length = len(img_paths)
                    i = 0
                    j = i + random.randrange(self.min_window_size, self.max_window_size)
                    while j < length:
                        img_seq = img_paths[i:j]
                        traj_seq = {'path': traj_dir, 'from': i, 'to': j}
                        preload_imgs.append(img_seq)
                        preload_traj.append(traj_seq)
                        i = j
                        j = i + random.randrange(self.min_window_size, self.max_window_size)
                preload_dict = {'imgs': preload_imgs, 'trajs': preload_traj}
                np.save(preload_file, preload_dict)
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.imgs += preload_dict.item()['imgs']
            self.trajs += preload_dict.item()['trajs']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data = dict()
        data['imgs'] = []
        seq_imgs = self.imgs[index]
        seq_trajs = self.trajs[index]
        data['seq_l'] = len(seq_imgs)
        for path in seq_imgs:
            data['imgs'].append(self.img_load(path))
        data['imgs'] = torch.tensor(np.array(data['imgs']))
        traj = np.load(os.path.join(seq_trajs['path'], 'data.npy'), allow_pickle=True)
        start = seq_trajs['from']
        end = seq_trajs['to']
        data['obs'] = torch.tensor(traj.item()['obs'][start:end])
        data['acts'] = torch.tensor(traj.item()['acts'][start:end])

        return data

    @staticmethod
    def img_load(path):
        img = Image.open(path)
        img_np = np.array(img)
        return img_np
