#!/usr/bin/env python
import os
import argparse
from run_train import Trainer
from arguments.train import parser as train_parser

def main():
    """
    Run the training process for the diffusion model
    """
    # Parse command line arguments
    args = train_parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(args)
    
    # Run training
    trainer.train()
    
    print(f"Training completed. Model saved in: {trainer.experiment_folder}")
    print(f"Best validation loss: {trainer.best_optimization_loss}")

if __name__ == "__main__":
    main() 