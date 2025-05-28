import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset import hw4_dataset
from model import PromptIR
from pathlib import Path


class Tester():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.dataset_path = args.dataset_path
        self.img_size = args.output_img_size
        self.num_workers = args.num_workers
        self.device = args.device
        self.seed = args.seed
        self.ckpt_path = args.ckpt_path
        self.save_img_dir = args.save_img_dir

        self.test_set = hw4_dataset(root_path=Path(self.dataset_path),
                                           mode="test",
                                           output_img_size=self.img_size,)

        self.test_loader = DataLoader(self.test_set,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=self.num_workers)

        self.model = PromptIR(decoder=True).to(self.device)
        self.load_ckpt(self.ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path)['model'])
        print(f'checkpoint: {ckpt_path}')

    def save_img(self, img_name, output: torch.tensor):
        img_name = img_name[0]
        output = output.detach().cpu()[0].numpy()

        arr = np.clip(output * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        pil_image = Image.fromarray(arr)
        pil_image.save(f"{self.save_img_dir}/{img_name}.png")

    @torch.no_grad()
    def test(self):
        self.model.eval()
        pbar = tqdm(self.test_loader, desc='Test', ncols=120)
        for img_name, img in pbar:
            img = img.to(self.device)
            output = self.model(img)
            self.save_img(img_name, output)
            pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=24)
    parser.add_argument('--output-img-size', type=int, default=64)
    parser.add_argument('--dataset_path', '-ds', type=str,
                        default='../hw4_realse_dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--save-img-dir', type=str, default='./img')
    parser.add_argument('--seed', type=int, default=110550074)
    args = parser.parse_args()

    os.makedirs(args.save_img_dir, exist_ok=True)
    Tester(args).test()
