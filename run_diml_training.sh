#!/bin/bash

# Run training using the DIML dataset in the datafolder directory
python train_diffusion.py \
  --save-dir ./experiments \
  --dataset DIML \
  --data-dir ./datafolder \
  --batch-size 8 \
  --num-epochs 1000 \
  --feature-extractor UNet \
  --Npre 8000 \
  --Ntrain 1024 \
  --use-transformer \
  --save-model both 