from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms as T
import numpy as np
import random


class hw4_dataset(Dataset):
    def __init__(self, root_path: Path, mode, output_img_size, shuffle_list: list = None):
        self.root_path = root_path
        self.mode = mode
        assert mode in ['train', 'valid',
                        'test'], f"{mode} should be 'train' or 'test'."
        self.output_img_size = output_img_size
        self.shuffle_list = shuffle_list
        self.transform = self.get_transformation()

        if self.mode in ['train', 'valid']:
            self.img_dir_path = self.root_path / 'train'
            self.clean_imgs = self.get_img_list(self.img_dir_path / 'clean')
            self.degraded_imgs = self.get_img_list(
                self.img_dir_path / 'degraded')
        else:
            self.img_dir_path = self.root_path / 'test'
            self.degraded_imgs = self.get_img_list(
                self.img_dir_path / 'degraded')

    def get_img_list(self, img_dir: Path):
        a = sorted(img_dir.glob('*.png'))

        if self.shuffle_list is None:
            return a

        if self.mode == 'train':
            return [img for img, keep in zip(a, self.shuffle_list) if keep]
        elif self.mode == 'valid':
            return [img for img, keep in zip(a, self.shuffle_list) if not keep]
        else:
            return a

    def get_transformation(self):
        if self.mode in ['train']:
            transformation = T.Compose(
                [T.RandomCrop(self.output_img_size), T.ToTensor()])
        elif self.mode in ['valid']:
            transformation = T.Compose([T.ToTensor()])
        else:
            transformation = T.Compose([T.ToTensor()])

        return transformation

    def __len__(self):
        return len(self.degraded_imgs)

    def __getitem__(self, idx):
        seed = np.random.randint(2147483647)
        if self.mode in ['train', 'valid']:
            degraded_img_path = self.degraded_imgs[idx]
            type_, num_ = degraded_img_path.stem.split('-')

            clean_img_path = self.img_dir_path / \
                'clean' / f'{type_}_clean-{num_}.png'
            if clean_img_path not in self.clean_imgs:
                raise ValueError('Not found clean image')

            degraded_img = Image.open(degraded_img_path).convert('RGB')
            clean_img = Image.open(clean_img_path).convert('RGB')

            if self.transform:
                random.seed(seed)
                torch.manual_seed(seed)
                degraded_img = self.transform(degraded_img)
                random.seed(seed)
                torch.manual_seed(seed)
                clean_img = self.transform(clean_img)

            return degraded_img, clean_img
        else:
            degraded_img_path = self.degraded_imgs[idx]

            degraded_img = Image.open(degraded_img_path).convert('RGB')
            if self.transform:
                degraded_img = self.transform(degraded_img)

            img_name = degraded_img_path.stem
            return img_name, degraded_img
