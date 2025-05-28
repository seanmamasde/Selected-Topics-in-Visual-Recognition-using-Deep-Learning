import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import hw4_dataset
from model import PromptIR
from pathlib import Path


def psnr_term(output: torch.Tensor,
              target: torch.Tensor,
              max_val: float = 1.0,
              eps: float = 1e-10) -> torch.Tensor:
    """
    Compute a PSNR-based loss term: negative mean PSNR over the batch.
    PSNR = 10 * log10(max_val^2 / MSE), so we return -PSNR.
    """
    # per-pixel MSE, then mean per image
    mse = F.mse_loss(output, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    # PSNR per image
    psnr = 10.0 * torch.log10(max_val**2 / mse.clamp(min=eps))
    # return negative mean PSNR
    return -psnr.mean()


class Training:
    def __init__(self, args):
        # Basic settings
        self.epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.dataset_path = args.dataset_path
        self.img_size = args.output_img_size
        self.num_workers = args.num_workers
        self.device = torch.device(args.device)
        self.train_ratio = args.train_ratio
        self.seed = args.seed
        self.lr = args.lr
        self.ckpt_dir = args.save_ckpt_dir
        self.save_img_dir = args.save_img_dir
        self.save_frequency = args.save_frequency
        self.resume_path = args.resume_from

        # Mixed-loss weights
        self.alpha = args.weight_l1
        self.beta = args.weight_psnr

        # Data loaders
        shuffle_list = self.generate_shuffle_list()
        self.train_set = hw4_dataset(
            root_path=Path(self.dataset_path),
            mode="train",
            output_img_size=self.img_size,
            shuffle_list=shuffle_list,
        )
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_set = hw4_dataset(
            root_path=Path(self.dataset_path),
            mode="valid",
            output_img_size=self.img_size,
            shuffle_list=shuffle_list,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=2,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Model, optimizer, scheduler
        self.model = PromptIR(decoder=True).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        # Loss
        self.l1_loss = nn.L1Loss()

        # Optionally resume
        self.start_epoch = 0
        if self.resume_path:
            print(f"Resuming from checkpoint: {self.resume_path}")
            ckpt = torch.load(self.resume_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt.get('epoch', 0) + 1
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr

    def generate_shuffle_list(self):
        random.seed(self.seed)
        total = 3200
        true_count = int(total * self.train_ratio)
        false_count = total - true_count
        bool_list = [True] * true_count + [False] * false_count
        random.shuffle(bool_list)
        return bool_list

    def save_ckpt(self, epoch: int):
        save_path = os.path.join(self.ckpt_dir, f"ckpt_{epoch}.pth")
        print(f"Saving checkpoint to {save_path}")
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_path)

    @torch.no_grad()
    def eval(self) -> tuple:
        self.model.eval()
        losses, psnrs = [], []
        for img, gt in self.val_loader:
            img, gt = img.to(self.device), gt.to(self.device)
            out = self.model(img)
            # L1 loss
            l1 = self.l1_loss(out, gt)
            # PSNR term
            psnr_loss = psnr_term(out, gt)
            # accumulate
            losses.append(l1.item())
            # convert back to positive PSNR for reporting
            psnrs.append((-psnr_loss).item())
        avg_l1 = float(np.mean(losses))
        avg_psnr = float(np.mean(psnrs))
        return avg_l1, avg_psnr

    def train_one_epoch(self) -> float:
        self.model.train()
        running_losses = []
        for img, gt in self.train_loader:
            img, gt = img.to(self.device), gt.to(self.device)
            out = self.model(img)
            # compute mixed loss
            loss_l1 = self.l1_loss(out, gt)
            loss_psnr = psnr_term(out, gt)
            loss = self.alpha * loss_l1 + self.beta * loss_psnr
            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_losses.append(loss.item())
        return float(np.mean(running_losses))

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch()
            print(
                f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Learning Rate: {self.optimizer.param_groups[0]['lr']:.1e}")
            # evaluate & scheduler step
            if (epoch + 1) % self.save_frequency == 0 or epoch == self.epochs-1:
                val_l1, val_psnr = self.eval()
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Val L1: {val_l1:.4f}, "
                    f"Val PSNR: {val_psnr:.2f} dB"
                )
                # reduce lr on plateau using validation L1 loss
                self.lr_scheduler.step(val_l1)
                self.save_ckpt(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--num-epochs', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--num-workers', type=int, default=24)
    parser.add_argument('--output-img-size', type=int, default=64)
    parser.add_argument('--dataset_path', type=str,
                        default='../hw4_realse_dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-ckpt-dir', type=str, default='./ckpts')
    parser.add_argument('--save-img-dir', type=str, default='./img')
    parser.add_argument('--save-frequency', type=int, default=25)
    parser.add_argument('--seed', type=int, default=110550074)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--weight-l1', type=float, default=1.0,
                        help='weight for L1 loss')
    parser.add_argument('--weight-psnr', type=float, default=0.01,
                        help='weight for PSNR loss term')
    parser.add_argument('--resume-from', type=str, default='',
                        help='path to checkpoint for resuming')
    args = parser.parse_args()

    os.makedirs(args.save_ckpt_dir, exist_ok=True)
    os.makedirs(args.save_img_dir, exist_ok=True)

    Training(args).run()
