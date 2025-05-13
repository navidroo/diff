import os
import argparse
from collections import defaultdict
import time

import torch
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import eval_parser
from model import GADBase
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from utils import to_cuda

from losses import get_loss
from frequency_losses import get_frequency_loss, FocalFrequencyLoss, LogFocalFrequencyLoss
import time


class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloader = self.get_dataloader(args)
        
        self.model = GADBase(args.feature_extractor, Npre=args.Npre, Ntrain=args.Ntrain)
        self.resume(path=args.checkpoint)
        self.model.cuda().eval()
        
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
        
        # Print the loss type being used for evaluation
        print(f"Using loss type for evaluation: {args.loss}")
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
            print("Using Automatic Mixed Precision (AMP) for evaluation")

    def evaluate(self):
        test_stats = defaultdict(float)
        num_samples = 0 

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloader, leave=False):
                sample = to_cuda(sample)
                
                # Use AMP autocast if enabled
                with torch.cuda.amp.autocast() if self.args.use_amp else torch.no_grad():
                    output = self.model(sample)
                    
                    # Calculate the primary loss based on the selected loss type
                    if self.args.loss == 'l1':
                        _, loss_dict = get_loss(output, sample)
                    elif self.args.loss == 'frequency':
                        _, loss_dict = get_frequency_loss(output, sample, 
                                                       alpha=self.args.freq_alpha, 
                                                       beta=self.args.freq_beta,
                                                       phase_weight=self.args.phase_weight,
                                                       use_log_focal=self.args.use_log_focal,
                                                       focal_gamma=self.args.focal_gamma,
                                                       focal_lambda=self.args.focal_lambda)
                    elif self.args.loss == 'hybrid':
                        # Compute both losses
                        _, l1_loss_dict = get_loss(output, sample)
                        freq_loss, freq_loss_dict = get_frequency_loss(output, sample, 
                                                                     alpha=self.args.freq_alpha, 
                                                                     beta=self.args.freq_beta,
                                                                     phase_weight=self.args.phase_weight,
                                                                     use_log_focal=self.args.use_log_focal,
                                                                     focal_gamma=self.args.focal_gamma,
                                                                     focal_lambda=self.args.focal_lambda)
                        
                        # Combine losses with weighting
                        w = self.args.hybrid_weight
                        hybrid_loss = w * freq_loss + (1-w) * l1_loss_dict['optimization_loss']
                        
                        # Combine the loss dicts
                        loss_dict = freq_loss_dict.copy()
                        loss_dict['hybrid_loss'] = hybrid_loss.detach().item()
                        loss_dict['optimization_loss'] = hybrid_loss.detach().item()
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
                        combined_loss = (1 - w_spectral) * loss_dict['optimization_loss'] + w_spectral * spectral_loss
                        
                        # Update loss for reporting
                        loss_dict['combined_loss'] = combined_loss.detach().item()
                        loss_dict['optimization_loss'] = combined_loss.detach().item()  # For consistent tracking

                for key in loss_dict:
                    test_stats[key] += loss_dict[key]

        return {k: v / len(self.dataloader) for k, v in test_stats.items()}

    @staticmethod
    def get_dataloader(args: argparse.Namespace):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': 0,
            'do_horizontal_flip': False,
            'crop_valid': True,
            'crop_deterministic': True,
            'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        if args.dataset == 'DIML':
            # depth_transform = Normalize([2749.64], [1154.29])
            depth_transform = Normalize([0.0], [1154.29])
            dataset = DIMLDataset(os.path.join(args.data_dir, 'DIML'), **data_args, split='test',
                                  depth_transform=depth_transform)
        elif args.dataset == 'Middlebury':
            # depth_transform = Normalize([2296.78], [1122.7])
            depth_transform = Normalize([0.0], [1122.7])
            dataset = MiddleburyDataset(os.path.join(args.data_dir, 'Middlebury'), **data_args, split='test',
                                        depth_transform=depth_transform)
        elif args.dataset == 'NYUv2':
            # depth_transform = Normalize([2796.32], [1386.05])
            depth_transform = Normalize([0.0], [1386.05])
            dataset = NYUv2Dataset(os.path.join(args.data_dir, 'NYU Depth v2'), **data_args, split='test',
                                   depth_transform=depth_transform)
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            model_dict = checkpoint['model']
            model_dict.pop('logk2', None) # in case of using the old codebase, pop unneccesary keys
            model_dict.pop('mean_guide', None)
            model_dict.pop('std_guide', None)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(checkpoint)
        print(f'Checkpoint \'{path}\' loaded.')

    def transfer(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = eval_parser.parse_args()
    print(eval_parser.format_values())

    evaluator = Evaluator(args)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    # de-standardize losses and convert to cm (cm^2, respectively)
    std = evaluator.dataloader.dataset.depth_transform.std[0]
    stats['l1_loss'] = 0.1 * std * stats['l1_loss']
    stats['mse_loss'] = 0.01 * std**2 * stats['mse_loss']
    
    # Also report frequency and spectral losses if available
    if 'freq_loss' in stats:
        print("Frequency Loss:", stats['freq_loss'])
        if 'mag_loss' in stats:
            print("Magnitude Loss:", stats['mag_loss'])
        if 'phase_loss' in stats:
            print("Phase Loss:", stats['phase_loss'])
    
    if 'spectral_loss' in stats:
        print("Spectral Loss:", stats['spectral_loss'])
        if 'spectral_mag_loss' in stats:
            print("Spectral Magnitude Loss:", stats['spectral_mag_loss'])
        if 'spectral_phase_loss' in stats:
            print("Spectral Phase Loss:", stats['spectral_phase_loss'])
    
    # Also report hybrid loss if available
    if 'hybrid_loss' in stats:
        print("Hybrid Loss:", stats['hybrid_loss'])
        
    # Report combined loss if spectral loss was used
    if 'combined_loss' in stats:
        print("Combined Loss:", stats['combined_loss'])

    print('Evaluation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(stats)