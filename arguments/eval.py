import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

# general
parser.add_argument('--dataset', type=str, help='Name of the dataset', required=True)
parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--in-memory', action='store_true', default=False, help='Hold data in memory during training')
parser.add_argument('--num-workers', type=int, default=8, help='Number of dataloader worker processes')

# data
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=8, help='Scaling factor')

# training
parser.add_argument('--loss', default='l1', type=str, choices=['l1', 'frequency', 'hybrid', 'curriculum'])
parser.add_argument('--use_spectral_loss', action='store_true', default=False, help='Use spectral loss in addition to the primary loss')
parser.add_argument('--spectral_loss_weight', type=float, default=0.5, help='Weight for spectral loss when used with --use_spectral_loss (0.0-1.0)')

# frequency loss parameters
parser.add_argument('--freq-alpha', type=float, default=1.0, help='Weight for magnitude component in frequency loss')
parser.add_argument('--freq-beta', type=float, default=1.0, help='Weight for phase component in frequency loss')
parser.add_argument('--phase-weight', type=float, default=0.5, help='Legacy parameter: Weight for phase consistency in frequency loss (0.0 = magnitude only, 1.0 = phase only)')
parser.add_argument('--use-log-focal', action='store_true', default=False, help='Use logarithmic-based focal weighting instead of sigmoid-based')
parser.add_argument('--focal-lambda', type=float, default=0.5, help='Controls strength of sigmoid-based focal weighting (higher = stronger effect)')
parser.add_argument('--focal-gamma', type=float, default=2.0, help='Controls strength of log-based focal weighting (higher = stronger effect)')

# New parameters
parser.add_argument('--magnitude-only', action='store_true', default=False, help='Use only magnitude component of frequency loss (no phase)')
parser.add_argument('--disable-focal-weight', action='store_true', default=False, help='Disable focal weighting in frequency loss')
parser.add_argument('--fft-norm', type=str, default='ortho', choices=['ortho', 'forward', 'backward', None], help='Normalization method for FFT')
parser.add_argument('--initial-spectral-weight', type=float, default=0.01, help='Initial weight for spectral loss in hybrid/curriculum mode')
parser.add_argument('--final-spectral-weight', type=float, default=0.3, help='Final weight for spectral loss in curriculum mode')
parser.add_argument('--hybrid-weight', type=float, default=0.5, help='Weight for frequency loss in hybrid loss (freq*w + l1*(1-w))')

parser.add_argument('--use_amp', action='store_true', default=False, help='Use Automatic Mixed Precision (AMP) for evaluation')

# model
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--model-type', type=str, default='GADBase', help='Model type')
parser.add_argument('--feature-extractor', type=str, default='UNet', help="Feature extractor for edge potentials. 'none' for the unlearned version.")
parser.add_argument('--Npre', type=int, default=8000, help='N learned iterations, but without gradients')
parser.add_argument('--Ntrain', type=int, default=1024, help='N learned iterations with gradients')