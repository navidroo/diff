# Diffusion Model Training

This document explains how to run training for the diffusion model and how models are saved.

## Running Training

There are two ways to run training:

### 1. Using a configuration file

```bash
python train_diffusion.py --config configs/training_config.yaml
```

### 2. Using command line arguments

```bash
python train_diffusion.py \
  --save-dir ./experiments \
  --dataset NYUv2 \
  --data-dir ./data \
  --batch-size 8 \
  --num-epochs 1000 \
  --feature-extractor UNet \
  --Npre 8000 \
  --Ntrain 1024 \
  --use-transformer
```

## Key Training Parameters

- `--save-dir`: Directory where models and logs will be saved
- `--dataset`: Dataset to use (NYUv2, Middlebury, or DIML)
- `--data-dir`: Root directory containing the dataset
- `--batch-size`: Training batch size
- `--num-epochs`: Number of training epochs
- `--save-model`: Model saving strategy ('best', 'last', 'both', or 'no')
- `--resume`: Path to checkpoint to resume training from

For a complete list of parameters, see `arguments/train.py`.

## How Models Are Saved

The training script saves models based on the `--save-model` parameter:

- When `--save-model` is set to `best` (default): Only the model with the best validation loss is saved as `best_model.pth`
- When `--save-model` is set to `last`: Only the most recent model is saved as `last_model.pth`
- When `--save-model` is set to `both`: Both the best model and the most recent model are saved
- When `--save-model` is set to `no`: No models are saved

### Model Checkpoints

Model checkpoints contain:
- Model state dictionary
- Optimizer state (for resuming training)
- Learning rate scheduler state
- Current epoch and iteration

### Best Model Selection

The best model is determined by the validation loss (specifically the 'optimization_loss'). During validation, which happens every `--val-every-n-epochs` epochs, the validation loss is calculated. If this loss is lower than the previous best, the model is saved as `best_model.pth`.

Models are saved in a timestamped experiment folder inside the specified `--save-dir` directory.

## Resuming Training

To resume training from a saved checkpoint:

```bash
python train_diffusion.py --config configs/training_config.yaml --resume /path/to/checkpoint.pth
``` 