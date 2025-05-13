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
from contextlib import nullcontext

from arguments import train_parser
from model import GADBase
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from losses import get_loss
from frequency_losses import get_frequency_loss, FocalFrequencyLoss, LogFocalFrequencyLoss
from utils import new_log, to_cuda, seed_all

# import nvidia_smi
# nvidia_smi.nvmlInit()

# Try to import wandb if needed
try:
    import wandb
except ImportError:
    pass

class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_wandb = self.args.wandb
        
        # Initialize AMP scaler if using mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

        self.dataloaders = self.get_dataloaders(args)
        
        seed_all(args.seed)

        self.model = GADBase( 
            args.feature_extractor, 
            Npre=args.Npre,
            Ntrain=args.Ntrain, 
        ).cuda()

        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, args.dataset), args)
        self.args.experiment_folder = self.experiment_folder

        if self.use_wandb:
            wandb.init(project=args.wandb_project, dir=self.experiment_folder)
            wandb.config.update(self.args)
            self.writer = None
        # else:
            # self.writer = SummaryWriter(log_dir=self.experiment_folder)

        if not args.no_opt:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        else:
            self.optimizer = None
            self.scheduler = None

        # Initialize spectral loss if needed
        if args.use_spectral_loss:
            try:
                self.spectral_loss_fn = FocalFrequencyLoss(
                    alpha=args.freq_alpha,
                    beta=args.freq_beta,
                    focal_lambda=args.focal_lambda
                ).cuda() if not args.use_log_focal else LogFocalFrequencyLoss(
                    alpha=args.freq_alpha,
                    beta=args.freq_beta,
                    focal_gamma=args.focal_gamma
                ).cuda()
                print(f"Initialized spectral loss: {'LogFocalFrequencyLoss' if args.use_log_focal else 'FocalFrequencyLoss'}")
            except Exception as e:
                print(f"Error initializing spectral loss: {e}")
                self.spectral_loss_fn = None
        else:
            self.spectral_loss_fn = None

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        if args.resume is not None:
            self.resume(path=args.resume)

        # Print the loss type being used
        print(f"Using loss type: {args.loss}")
        if args.use_spectral_loss:
            print(f"Additional spectral loss enabled with weight: {args.spectral_loss_weight}")
            print(f"Spectral loss parameters - Alpha: {args.freq_alpha}, Beta: {args.freq_beta}")
            if args.use_log_focal:
                print(f"Using logarithmic focal weighting with gamma: {args.focal_gamma}")
            else:
                print(f"Using sigmoid focal weighting with lambda: {args.focal_lambda}")
        elif args.loss in ['frequency', 'hybrid']:
            print(f"Frequency loss parameters - Alpha: {args.freq_alpha}, Beta: {args.freq_beta}")
            if args.use_log_focal:
                print(f"Using logarithmic focal weighting with gamma: {args.focal_gamma}")
            else:
                print(f"Using sigmoid focal weighting with lambda: {args.focal_lambda}")
            if args.loss == 'hybrid':
                print(f"Hybrid loss weight: {args.hybrid_weight}")
        
        if args.use_amp:
            print("Using Automatic Mixed Precision (AMP) training")

    def __del__(self):
        if not self.use_wandb:
            self.writer.close()

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    if not args.no_opt:
                        self.scheduler.step()
                        if self.use_wandb:
                            wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)
                        else:
                            self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1

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

                if not args.no_opt:
                    self.optimizer.zero_grad()

                # Use AMP autocast if enabled - but don't use torch.no_grad for training!
                with torch.cuda.amp.autocast() if self.args.use_amp else nullcontext():
                    # Pass context to the model
                    output = self.model(sample, train=True)
                    
                    # Calculate the primary loss based on the selected loss type
                    if self.args.loss == 'l1':
                        loss, loss_dict = get_loss(output, sample)
                    elif self.args.loss == 'frequency':
                        loss, loss_dict = get_frequency_loss(output, sample, 
                                                           alpha=self.args.freq_alpha, 
                                                           beta=self.args.freq_beta,
                                                           phase_weight=self.args.phase_weight,
                                                           use_log_focal=self.args.use_log_focal,
                                                           focal_gamma=self.args.focal_gamma,
                                                           focal_lambda=self.args.focal_lambda)
                    elif self.args.loss == 'hybrid':
                        # Compute both losses
                        l1_loss, l1_loss_dict = get_loss(output, sample)
                        freq_loss, freq_loss_dict = get_frequency_loss(output, sample, 
                                                                     alpha=self.args.freq_alpha, 
                                                                     beta=self.args.freq_beta,
                                                                     phase_weight=self.args.phase_weight,
                                                                     use_log_focal=self.args.use_log_focal,
                                                                     focal_gamma=self.args.focal_gamma,
                                                                     focal_lambda=self.args.focal_lambda)
                        
                        # Combine losses with weighting
                        w = self.args.hybrid_weight
                        loss = w * freq_loss + (1-w) * l1_loss  # Use the raw tensor with gradients
                        
                        # Combine the loss dicts
                        loss_dict = freq_loss_dict.copy()
                        loss_dict['hybrid_loss'] = loss.detach().item()
                        loss_dict['optimization_loss'] = loss.detach().item()
                    else:
                        raise ValueError(f"Unknown loss type: {self.args.loss}")
                    
                    # Add spectral loss if enabled (applied to the final adjusted high-res depth prediction)
                    if self.args.use_spectral_loss and self.spectral_loss_fn is not None:
                        y_pred = output['y_pred']
                        y, mask_hr = sample['y'], sample['mask_hr']
                        
                        # Calculate spectral loss
                        spectral_loss, mag_loss, phase_loss = self.spectral_loss_fn(y_pred, y, mask_hr)
                        
                        # Add spectral loss components to the loss dict
                        loss_dict['spectral_loss'] = spectral_loss.detach().item()
                        loss_dict['spectral_mag_loss'] = mag_loss.detach().item()
                        loss_dict['spectral_phase_loss'] = phase_loss.detach().item()
                        
                        # Combine with primary loss
                        w_spectral = self.args.spectral_loss_weight
                        combined_loss = (1 - w_spectral) * loss + w_spectral * spectral_loss
                        
                        # Update loss for backpropagation
                        loss = combined_loss
                        loss_dict['combined_loss'] = loss.detach().item()
                        loss_dict['optimization_loss'] = loss.detach().item()  # For consistent tracking

                if torch.isnan(loss):
                    raise Exception("detected NaN loss..")
                    
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

                if self.epoch > 0 or not self.args.skip_first:
                    if not args.no_opt:
                        if self.args.use_amp:
                            # Use AMP-aware backward and optimizer step
                            self.scaler.scale(loss).backward()
                            
                            if self.args.gradient_clip > 0.:
                                self.scaler.unscale_(self.optimizer)
                                clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                                
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # Standard backward and optimizer step
                            loss.backward()
                            
                            if self.args.gradient_clip > 0.:
                                clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                                
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
                    else:
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

                # Use AMP autocast if enabled (no gradients needed in validation)
                with torch.cuda.amp.autocast() if self.args.use_amp else nullcontext():
                    output = self.model(sample)
                    
                    # Calculate the primary loss based on the selected loss type
                    if self.args.loss == 'l1':
                        loss, loss_dict = get_loss(output, sample)
                    elif self.args.loss == 'frequency':
                        loss, loss_dict = get_frequency_loss(output, sample, 
                                                           alpha=self.args.freq_alpha, 
                                                           beta=self.args.freq_beta,
                                                           phase_weight=self.args.phase_weight,
                                                           use_log_focal=self.args.use_log_focal,
                                                           focal_gamma=self.args.focal_gamma,
                                                           focal_lambda=self.args.focal_lambda)
                    elif self.args.loss == 'hybrid':
                        # Compute both losses
                        l1_loss, l1_loss_dict = get_loss(output, sample)
                        freq_loss, freq_loss_dict = get_frequency_loss(output, sample, 
                                                                     alpha=self.args.freq_alpha, 
                                                                     beta=self.args.freq_beta,
                                                                     phase_weight=self.args.phase_weight,
                                                                     use_log_focal=self.args.use_log_focal,
                                                                     focal_gamma=self.args.focal_gamma,
                                                                     focal_lambda=self.args.focal_lambda)
                        
                        # Combine losses with weighting
                        w = self.args.hybrid_weight
                        loss = w * freq_loss + (1-w) * l1_loss  # Use the raw tensor with gradients
                        
                        # Combine the loss dicts
                        loss_dict = freq_loss_dict.copy()
                        loss_dict['hybrid_loss'] = loss.detach().item()
                        loss_dict['optimization_loss'] = loss.detach().item()
                    else:
                        raise ValueError(f"Unknown loss type: {self.args.loss}")
                    
                    # Add spectral loss if enabled (applied to the final adjusted high-res depth prediction)
                    if self.args.use_spectral_loss and self.spectral_loss_fn is not None:
                        y_pred = output['y_pred']
                        y, mask_hr = sample['y'], sample['mask_hr']
                        
                        # Calculate spectral loss
                        spectral_loss, mag_loss, phase_loss = self.spectral_loss_fn(y_pred, y, mask_hr)
                        
                        # Add spectral loss components to the loss dict
                        loss_dict['spectral_loss'] = spectral_loss.detach().item()
                        loss_dict['spectral_mag_loss'] = mag_loss.detach().item()
                        loss_dict['spectral_phase_loss'] = phase_loss.detach().item()
                        
                        # Combine with primary loss
                        w_spectral = self.args.spectral_loss_weight
                        combined_loss = (1 - w_spectral) * loss + w_spectral * spectral_loss
                        
                        # Update loss for backpropagation
                        loss = combined_loss
                        loss_dict['combined_loss'] = loss.detach().item()
                        loss_dict['optimization_loss'] = loss.detach().item()  # For consistent tracking

                for key in loss_dict:
                    self.val_stats[key] +=  loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            if self.use_wandb:
                wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            else:
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
        save_dict = {
            'model': self.model.state_dict(),
            'epoch': self.epoch + 1,
            'iter': self.iter
        }
        
        if not args.no_opt:
            save_dict.update({
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            })
            
        if self.args.use_amp and self.scaler is not None:
            save_dict['amp_scaler'] = self.scaler.state_dict()
            
        torch.save(save_dict, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        
        if not args.no_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        if 'amp_scaler' in checkpoint and self.args.use_amp and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['amp_scaler'])
            
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
