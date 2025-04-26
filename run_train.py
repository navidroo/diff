import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch import is_tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm

from arguments import train_parser
from model import GADBase
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from losses import get_loss
from utils import new_log, to_cuda, seed_all

# import nvidia_smi
# nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_wandb = self.args.wandb

        self.dataloaders = self.get_dataloaders(args)
        
        seed_all(args.seed)

        self.model = GADBase( 
            args.feature_extractor, 
            Npre=args.Npre,
            Ntrain=args.Ntrain,
            use_transformer=args.use_transformer,
            transformer_blocks=args.transformer_blocks,
            transformer_heads=args.transformer_heads
        ).cuda()

        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, args.dataset), args)
        self.args.experiment_folder = self.experiment_folder

        if self.use_wandb:
            wandb.init(project=self.args.wandb_project, dir=self.experiment_folder)
            wandb.config.update(self.args)
            self.writer = None
        else:
            # Initialize TensorBoard writer or set to None
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.experiment_folder)
            except ImportError:
                print("TensorBoard not available, not logging training metrics.")
                self.writer = None

        if not self.args.no_opt:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_step, gamma=self.args.lr_gamma)
        else:
            self.optimizer = None
            self.scheduler = None

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf
        
        # Initialize early stopping variables
        self.patience = self.args.early_stopping_patience
        self.patience_counter = 0
        self.early_stopping = False

        if self.args.resume is not None:
            self.resume(path=self.args.resume)

    def __del__(self):
        if not self.use_wandb and hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    prev_best = self.best_optimization_loss
                    self.validate()
                    
                    # Check for early stopping
                    if self.val_stats['optimization_loss'] >= prev_best:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            print(f"\nEarly stopping triggered! No improvement for {self.patience} validation checks.")
                            print(f"Best validation loss: {self.best_optimization_loss}")
                            self.early_stopping = True
                            break
                    else:
                        # Reset counter if validation loss improved
                        self.patience_counter = 0

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    if not self.args.no_opt:
                        self.scheduler.step()
                        if self.use_wandb:
                            wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)
                        elif self.writer is not None:
                            self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1
                
                # Update progress bar with early stopping info
                tnr.set_postfix(
                    training_loss=self.train_stats['optimization_loss'] if 'optimization_loss' in self.train_stats else np.nan,
                    validation_loss=self.val_stats['optimization_loss'],
                    best_validation_loss=self.best_optimization_loss,
                    patience=f"{self.patience_counter}/{self.patience}"
                )
                
                if self.early_stopping:
                    break

    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()
        
        # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # self.train_stats["gpu_used"] = info.used


        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                sample = to_cuda(sample)

                if not self.args.no_opt:
                    self.optimizer.zero_grad()

                output = self.model(sample, train=True)

                loss, loss_dict = get_loss(output, sample)

                if torch.isnan(loss):
                    raise Exception("detected NaN loss..")
                    
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

                if self.epoch > 0 or not self.args.skip_first:
                    if not self.args.no_opt:
                        loss.backward()

                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    if not self.args.no_opt:
                        self.optimizer.step()

                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss=self.val_stats['optimization_loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    if self.use_wandb:
                        wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    elif self.writer is not None:
                        for key in self.train_stats:
                            self.writer.add_scalar('train/' + key, self.train_stats[key], self.iter)

                    # reset metrics
                    self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders['val'], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)

                loss, loss_dict = get_loss(output, sample)

                for key in loss_dict:
                    self.val_stats[key] +=  loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            if self.use_wandb:
                wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            elif self.writer is not None:
                for key in self.val_stats:
                    self.writer.add_scalar('val/' + key, self.val_stats[key], self.epoch)

            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    @staticmethod
    def get_dataloaders(args):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': args.max_rotation,
            'do_horizontal_flip': not args.no_flip,
            'crop_valid': True,
            'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        phases = ('train', 'val')
        if args.dataset == 'Middlebury':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1122.7]) 
            datasets = {phase: MiddleburyDataset(os.path.join(args.data_dir, 'Middlebury'), **data_args, split=phase,
                        depth_transform=depth_transform, crop_deterministic=phase == 'val') for phase in phases}

        elif args.dataset == 'DIML':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1154.29])
            datasets = {phase: DIMLDataset(os.path.join(args.data_dir, 'DIML'), **data_args, split=phase,
                        depth_transform=depth_transform) for phase in phases}

        elif args.dataset == 'NYUv2':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1386.05])
            datasets = {phase: NYUv2Dataset(os.path.join(args.data_dir, 'NYU Depth v2'), **data_args, split=phase,
                        depth_transform=depth_transform) for phase in phases}
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return {phase: DataLoader(datasets[phase], batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=True, drop_last=False) for phase in phases}

    def save_model(self, prefix=''):
        if self.args.no_opt:
            torch.save({
                'model': self.model.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))
        else:
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if not self.args.no_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())

    if args.wandb:
        import wandb

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
