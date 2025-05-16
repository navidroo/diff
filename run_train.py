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
from torch.nn import functional as F

from arguments import train_parser
from model import GADBase
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from losses import get_loss
from frequency_losses import get_frequency_loss, FocalFrequencyLoss, LogFocalFrequencyLoss, MagnitudeFrequencyLoss
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
        if args.use_spectral_loss or args.loss in ['frequency', 'hybrid', 'curriculum']:
            try:
                if args.magnitude_only:
                    self.spectral_loss_fn = MagnitudeFrequencyLoss(
                        alpha=args.freq_alpha,
                        use_focal_weight=not args.disable_focal_weight,
                        focal_lambda=args.focal_lambda,
                        norm=args.fft_norm
                    ).cuda()
                    print(f"Initialized magnitude-only frequency loss")
                elif args.use_log_focal:
                    self.spectral_loss_fn = LogFocalFrequencyLoss(
                        alpha=args.freq_alpha,
                        beta=args.freq_beta,
                        focal_gamma=args.focal_gamma
                    ).cuda()
                    print(f"Initialized log-focal frequency loss")
                else:
                    self.spectral_loss_fn = FocalFrequencyLoss(
                        alpha=args.freq_alpha,
                        beta=args.freq_beta,
                        focal_lambda=args.focal_lambda
                    ).cuda()
                    print(f"Initialized sigmoid-focal frequency loss")
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
        if args.magnitude_only:
            print(f"Using magnitude-only frequency loss (no phase component)")
        
        if args.loss == 'curriculum':
            print(f"Curriculum learning: L1-only for {args.curriculum_delay_epochs} epochs, "
                  f"then ramping spectral loss from {args.initial_spectral_weight} to {args.final_spectral_weight} "
                  f"over {args.curriculum_ramp_epochs} epochs")
        
        if args.use_spectral_loss:
            print(f"Additional spectral loss enabled with weight: {args.spectral_loss_weight}")
        
        if args.use_spectral_loss or args.loss in ['frequency', 'hybrid', 'curriculum']:
            print(f"Spectral loss parameters - Alpha: {args.freq_alpha}" + 
                  (f", Beta: {args.freq_beta}" if not args.magnitude_only else ""))
            
            if args.use_log_focal and not args.disable_focal_weight:
                print(f"Using logarithmic focal weighting with gamma: {args.focal_gamma}")
            elif not args.disable_focal_weight:
                print(f"Using sigmoid focal weighting with lambda: {args.focal_lambda}")
            else:
                print("Focal weighting disabled")
                
            if args.loss == 'hybrid':
                print(f"Hybrid loss weight: {args.hybrid_weight}")
        
        if args.use_amp:
            print("Using Automatic Mixed Precision (AMP) training")

    def __del__(self):
        if not self.use_wandb:
            if hasattr(self, 'writer') and self.writer is not None:
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
                    if not self.args.no_opt:
                        self.scheduler.step()
                        if self.use_wandb:
                            wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr()[0])}, self.iter)
                        # else:
                        #     self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()[0]), self.epoch)

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

                if not self.args.no_opt:
                    self.optimizer.zero_grad()

                # Use AMP autocast if enabled - but don't use torch.no_grad for training!
                with torch.cuda.amp.autocast() if self.args.use_amp else nullcontext():
                    # Pass context to the model
                    output = self.model(sample, train=True)
                    
                    # Calculate L1 loss (needed for all cases)
                    l1_loss, l1_loss_dict = get_loss(output, sample)
                    
                    # Handle curriculum learning if selected
                    if self.args.loss == 'curriculum':
                        # Start with L1 only for specified number of epochs
                        if self.epoch < self.args.curriculum_delay_epochs:
                            # Only L1 loss during initial phase
                            loss = l1_loss
                            spectral_weight = 0.0
                        else:
                            # Calculate current spectral weight based on curriculum
                            if self.args.curriculum_ramp_epochs > 0 and \
                               self.epoch < (self.args.curriculum_delay_epochs + self.args.curriculum_ramp_epochs):
                                # Linear ramp from initial to final weight
                                progress = (self.epoch - self.args.curriculum_delay_epochs) / self.args.curriculum_ramp_epochs
                                spectral_weight = self.args.initial_spectral_weight + \
                                                 progress * (self.args.final_spectral_weight - self.args.initial_spectral_weight)
                            else:
                                # After ramp, use final weight
                                spectral_weight = self.args.final_spectral_weight
                            
                            # Calculate frequency loss with magnitude-only option
                            freq_loss, freq_loss_dict = get_frequency_loss(
                                output, sample, 
                                alpha=self.args.freq_alpha, 
                                beta=self.args.freq_beta,
                                phase_weight=self.args.phase_weight,
                                use_log_focal=self.args.use_log_focal,
                                focal_gamma=self.args.focal_gamma,
                                focal_lambda=self.args.focal_lambda,
                                magnitude_only=self.args.magnitude_only,
                                use_focal_weight=not self.args.disable_focal_weight,
                                norm=self.args.fft_norm
                            )
                            
                            # Combine losses with curriculum weight
                            loss = (1 - spectral_weight) * l1_loss + spectral_weight * freq_loss
                            
                            # Update loss dictionary
                            loss_dict = {**l1_loss_dict, **freq_loss_dict}
                            loss_dict['curriculum_spectral_weight'] = spectral_weight
                            loss_dict['optimization_loss'] = loss.detach().item()
                    
                    # Calculate the primary loss based on the selected loss type
                    elif self.args.loss == 'l1':
                        loss = l1_loss
                        loss_dict = l1_loss_dict
                        
                    elif self.args.loss == 'frequency':
                        loss, loss_dict = get_frequency_loss(
                            output, sample, 
                            alpha=self.args.freq_alpha, 
                            beta=self.args.freq_beta,
                            phase_weight=self.args.phase_weight,
                            use_log_focal=self.args.use_log_focal,
                            focal_gamma=self.args.focal_gamma,
                            focal_lambda=self.args.focal_lambda,
                            magnitude_only=self.args.magnitude_only,
                            use_focal_weight=not self.args.disable_focal_weight,
                            norm=self.args.fft_norm
                        )
                        
                    elif self.args.loss == 'hybrid':
                        # Compute frequency loss
                        freq_loss, freq_loss_dict = get_frequency_loss(
                            output, sample, 
                            alpha=self.args.freq_alpha, 
                            beta=self.args.freq_beta,
                            phase_weight=self.args.phase_weight,
                            use_log_focal=self.args.use_log_focal,
                            focal_gamma=self.args.focal_gamma,
                            focal_lambda=self.args.focal_lambda,
                            magnitude_only=self.args.magnitude_only,
                            use_focal_weight=not self.args.disable_focal_weight,
                            norm=self.args.fft_norm
                        )
                        
                        # Combine losses with weighting
                        w = self.args.hybrid_weight
                        loss = w * freq_loss + (1-w) * l1_loss
                        
                        # Combine the loss dicts
                        loss_dict = {**freq_loss_dict, **l1_loss_dict}
                        loss_dict['hybrid_loss'] = loss.detach().item()
                        loss_dict['optimization_loss'] = loss.detach().item()
                    else:
                        raise ValueError(f"Unknown loss type: {self.args.loss}")
                    
                    # Add spectral loss if enabled (applied to the final adjusted high-res depth prediction)
                    if self.args.use_spectral_loss and self.spectral_loss_fn is not None:
                        y_pred = output['y_pred']
                        y, mask_hr = sample['y'], sample['mask_hr']
                        
                        # Use the spectral loss with configured parameters
                        if isinstance(self.spectral_loss_fn, MagnitudeFrequencyLoss):
                            spectral_loss, spectral_mag_loss = self.spectral_loss_fn(y_pred, y, mask_hr)
                            spectral_phase_loss = torch.tensor(0.0, device=y_pred.device)
                        else:
                            spectral_loss, spectral_mag_loss, spectral_phase_loss = self.spectral_loss_fn(y_pred, y, mask_hr)
                        
                        # Add spectral loss components to the loss dict
                        loss_dict['spectral_loss'] = spectral_loss.detach().item()
                        loss_dict['spectral_mag_loss'] = spectral_mag_loss.detach().item()
                        if not isinstance(self.spectral_loss_fn, MagnitudeFrequencyLoss):
                            loss_dict['spectral_phase_loss'] = spectral_phase_loss.detach().item()
                        
                        # Combine with primary loss
                        w_spectral = self.args.spectral_loss_weight
                        combined_loss = (1 - w_spectral) * loss + w_spectral * spectral_loss
                        
                        # Update loss for reporting
                        loss_dict['combined_loss'] = combined_loss.detach().item()
                        loss_dict['original_loss'] = loss.detach().item()
                        loss = combined_loss
                        loss_dict['optimization_loss'] = loss.detach().item()

                    if torch.isnan(loss):
                        print("NaN loss detected, skipping batch")
                        continue
                    
                    for key in loss_dict:
                        self.train_stats[key] += loss_dict[key] if not torch.is_tensor(loss_dict[key]) else loss_dict[key].detach().cpu().item()

                    if self.epoch > 0 or not self.args.skip_first:
                        if not self.args.no_opt:
                            if self.args.use_amp:
                                # Use AMP's scaled_backward
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()

                        if self.args.gradient_clip > 0.:
                            if self.args.use_amp:
                                # First unscale the gradients
                                self.scaler.unscale_(self.optimizer)
                            clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                        if not self.args.no_opt:
                            if self.args.use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
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
                            if hasattr(self, 'writer') and self.writer is not None:
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

                # Use AMP autocast if enabled
                with torch.cuda.amp.autocast() if self.args.use_amp else torch.no_grad():
                    output = self.model(sample)
                    
                    # Calculate L1 loss (needed for all cases)
                    l1_loss, l1_loss_dict = get_loss(output, sample)
                    
                    # Handle curriculum learning if selected
                    if self.args.loss == 'curriculum':
                        # Calculate current spectral weight based on curriculum
                        if self.epoch < self.args.curriculum_delay_epochs:
                            # Only L1 loss during initial phase
                            loss = l1_loss
                            loss_dict = l1_loss_dict
                            loss_dict['curriculum_spectral_weight'] = 0.0
                        else:
                            # Calculate current spectral weight
                            if self.args.curriculum_ramp_epochs > 0 and \
                               self.epoch < (self.args.curriculum_delay_epochs + self.args.curriculum_ramp_epochs):
                                # Linear ramp from initial to final weight
                                progress = (self.epoch - self.args.curriculum_delay_epochs) / self.args.curriculum_ramp_epochs
                                spectral_weight = self.args.initial_spectral_weight + \
                                                progress * (self.args.final_spectral_weight - self.args.initial_spectral_weight)
                            else:
                                # After ramp, use final weight
                                spectral_weight = self.args.final_spectral_weight
                            
                            # Calculate frequency loss with magnitude-only option
                            freq_loss, freq_loss_dict = get_frequency_loss(
                                output, sample, 
                                alpha=self.args.freq_alpha, 
                                beta=self.args.freq_beta,
                                phase_weight=self.args.phase_weight,
                                use_log_focal=self.args.use_log_focal,
                                focal_gamma=self.args.focal_gamma,
                                focal_lambda=self.args.focal_lambda,
                                magnitude_only=self.args.magnitude_only,
                                use_focal_weight=not self.args.disable_focal_weight,
                                norm=self.args.fft_norm
                            )
                            
                            # Combine losses with curriculum weight
                            loss = (1 - spectral_weight) * l1_loss + spectral_weight * freq_loss
                            
                            # Update loss dictionary
                            loss_dict = {**l1_loss_dict, **freq_loss_dict}
                            loss_dict['curriculum_spectral_weight'] = spectral_weight
                            loss_dict['optimization_loss'] = loss.detach().item()
                    
                    # Calculate the primary loss based on the selected loss type
                    elif self.args.loss == 'l1':
                        loss = l1_loss
                        loss_dict = l1_loss_dict
                        
                    elif self.args.loss == 'frequency':
                        loss, loss_dict = get_frequency_loss(
                            output, sample, 
                            alpha=self.args.freq_alpha, 
                            beta=self.args.freq_beta,
                            phase_weight=self.args.phase_weight,
                            use_log_focal=self.args.use_log_focal,
                            focal_gamma=self.args.focal_gamma,
                            focal_lambda=self.args.focal_lambda,
                            magnitude_only=self.args.magnitude_only,
                            use_focal_weight=not self.args.disable_focal_weight,
                            norm=self.args.fft_norm
                        )
                        
                    elif self.args.loss == 'hybrid':
                        # Compute frequency loss
                        freq_loss, freq_loss_dict = get_frequency_loss(
                            output, sample, 
                            alpha=self.args.freq_alpha, 
                            beta=self.args.freq_beta,
                            phase_weight=self.args.phase_weight,
                            use_log_focal=self.args.use_log_focal,
                            focal_gamma=self.args.focal_gamma,
                            focal_lambda=self.args.focal_lambda,
                            magnitude_only=self.args.magnitude_only,
                            use_focal_weight=not self.args.disable_focal_weight,
                            norm=self.args.fft_norm
                        )
                        
                        # Combine losses with weighting
                        w = self.args.hybrid_weight
                        loss = w * freq_loss + (1-w) * l1_loss
                        
                        # Combine the loss dicts
                        loss_dict = {**freq_loss_dict, **l1_loss_dict}
                        loss_dict['hybrid_loss'] = loss.detach().item()
                        loss_dict['optimization_loss'] = loss.detach().item()
                    else:
                        raise ValueError(f"Unknown loss type: {self.args.loss}")
                    
                    # Add spectral loss if enabled
                    if self.args.use_spectral_loss and self.spectral_loss_fn is not None:
                        y_pred = output['y_pred']
                        y, mask_hr = sample['y'], sample['mask_hr']
                        
                        # Use the spectral loss with configured parameters
                        if isinstance(self.spectral_loss_fn, MagnitudeFrequencyLoss):
                            spectral_loss, spectral_mag_loss = self.spectral_loss_fn(y_pred, y, mask_hr)
                            spectral_phase_loss = torch.tensor(0.0, device=y_pred.device)
                        else:
                            spectral_loss, spectral_mag_loss, spectral_phase_loss = self.spectral_loss_fn(y_pred, y, mask_hr)
                        
                        # Add spectral loss components to the loss dict
                        loss_dict['spectral_loss'] = spectral_loss.detach().item()
                        loss_dict['spectral_mag_loss'] = spectral_mag_loss.detach().item()
                        if not isinstance(self.spectral_loss_fn, MagnitudeFrequencyLoss):
                            loss_dict['spectral_phase_loss'] = spectral_phase_loss.detach().item()
                        
                        # Combine with primary loss
                        w_spectral = self.args.spectral_loss_weight
                        combined_loss = (1 - w_spectral) * loss + w_spectral * spectral_loss
                        
                        # Update loss for reporting
                        loss_dict['combined_loss'] = combined_loss.detach().item()
                        loss_dict['original_loss'] = loss.detach().item()
                        loss = combined_loss
                        loss_dict['optimization_loss'] = loss.detach().item()

                    # Always calculate and log L1 and MSE for comparison with other models
                    if 'l1_loss' not in loss_dict:
                        y_pred = output['y_pred']
                        y, mask_hr = sample['y'], sample['mask_hr']
                        l1 = F.l1_loss(y_pred[mask_hr == 1.], y[mask_hr == 1.])
                        mse = F.mse_loss(y_pred[mask_hr == 1.], y[mask_hr == 1.])
                        loss_dict['l1_loss'] = l1.detach().item()
                        loss_dict['mse_loss'] = mse.detach().item()

                for key in loss_dict:
                    self.val_stats[key] += loss_dict[key] if not torch.is_tensor(loss_dict[key]) else loss_dict[key].detach().cpu().item()

            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            if self.use_wandb:
                wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            else:
                if hasattr(self, 'writer') and self.writer is not None:
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
        
        if not self.args.no_opt:
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
        
        if not self.args.no_opt:
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
