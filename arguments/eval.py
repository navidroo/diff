import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

# general
parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
parser.add_argument('--seed', type=int, default=12345, help='Random seed')

# data
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--in-memory', action='store_true', default=False, help='Hold data in memory during training')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=8, help='Scaling factor')

# model
parser.add_argument('--feature-extractor', type=str, default='UNet', help="Feature extractor for edge potentials. 'none' for the unlearned version.") 
parser.add_argument('--Npre', type=int, default=8000, help='N learned iterations, but without gradients')
parser.add_argument('--Ntrain', type=int, default=1024, help='N learned iterations with gradients')

# transformer options
parser.add_argument('--use-transformer', action='store_true', default=False, help='Enable transformer blocks for long-range dependencies')
parser.add_argument('--transformer-blocks', type=int, default=2, help='Number of transformer blocks')
parser.add_argument('--transformer-heads', type=int, default=8, help='Number of attention heads in transformer')