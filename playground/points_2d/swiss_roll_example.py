#!/usr/bin/env python3
"""
Swiss Roll Diffusion Model Example

This script demonstrates training and inference with a diffusion model on Swiss roll data.
It includes visualization of the training process and generation of inference GIF.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm

# Add the diffusion package to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from diffusion.denoisers import KarrasDenoiser
from diffusion.inference import KarrasHeun2Solver, KarrasNoiseSchedule
from diffusion.lightning import LightningDiffusion, TrainingConfig, InferenceConfig
from diffusion.training import (
    KarrasLossFn, 
    LogUniformSigmaSampler, 
    WEIGHTING_SCHEMES,
    EMAWarmupSchedule
)
from playground.points_2d.data import create_train_val_datasets
from playground.points_2d.model import PointDenoisingModel


class SwissRollDiffusionExample:
    """Complete example for training and inference with Swiss roll data."""
    
    def __init__(
        self,
        n_train: int = 2**12,
        n_val: int = 2**9,
        n_dims: int = 2,
        noise: float = 0.0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.n_train = n_train
        self.n_val = n_val
        self.n_dims = n_dims
        self.noise = noise
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Create datasets
        self.train_dataset, self.val_dataset = create_train_val_datasets(
            n_train=n_train, n_val=n_val, n_dims=n_dims, noise=noise
        )
        
        # Initialize model
        import pdb; pdb.set_trace()
        self.model = PointDenoisingModel(
            input_dim=n_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        # Wrap with Karras denoiser
        self.denoiser = KarrasDenoiser(
            model=self.model,
            input_shape=(n_dims,),
            output_shape=(n_dims,)
        )
        
        # Create output directory
        self.output_dir = Path("swiss_roll_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize lightning module
        self.lightning_module = None
        
    def setup_training_config(self) -> TrainingConfig:
        """Setup training configuration."""
        return TrainingConfig(
            # Loss function and weighting
            loss_fn=KarrasLossFn(),
            loss_weight_fn=WEIGHTING_SCHEMES["uniform"],
            sigma_sampler=LogUniformSigmaSampler(min_value=0.002, max_value=80.0),
            
            # EMA schedule
            ema_schedule=EMAWarmupSchedule(
                inv_gamma=1.0,
                power=1.0,
                min_value=0.0,
                max_value=0.9999,
                start_at=0,
                last_epoch=0,
            ),
            
            # Optimizer
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.01},
            
            # Learning rate scheduler
            lr_scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
            lr_scheduler_kwargs={"T_max": 100},
            lr_scheduler_interval="epoch",
        )
    
    def setup_inference_config(self) -> InferenceConfig:
        """Setup inference configuration."""
        from diffusion.inference import KarrasDiffEq
        
        return InferenceConfig(
            ode_builder=KarrasDiffEq,
            solver=KarrasHeun2Solver(),
            noise_schedule=KarrasNoiseSchedule(
                sigma_data=0.5,
                sigma_min=0.002,
                sigma_max=80.0,
                rho=7.0
            ),
            n_steps=50,
            return_trajectory=True
        )
        
    def train(self, epochs: int = 100, batch_size: int = 128, lr: float = 1e-4):
        """Train the diffusion model."""
        print(f"Training diffusion model for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        # Setup training configuration
        training_config = self.setup_training_config()
        training_config.optimizer_kwargs["lr"] = lr
        
        # Initialize lightning module
        self.lightning_module = LightningDiffusion(
            model=self.denoiser,
            training_config=training_config
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            log_every_n_steps=10,
            val_check_interval=0.25,
            enable_checkpointing=True,
            default_root_dir=str(self.output_dir),
        )
        
        # Train the model
        trainer.fit(
            model=self.lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Save the model
        torch.save(self.lightning_module.state_dict(), self.output_dir / "model.pt")
        print("Training completed!")
        
    def generate_samples(self, n_samples: int = 1000, n_steps: int = 50) -> np.ndarray:
        """Generate samples using the trained model."""
        print(f"Generating {n_samples} samples...")
        
        if self.lightning_module is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Setup inference configuration
        inference_config = self.setup_inference_config()
        inference_config.n_steps = n_steps
        self.lightning_module.setup_inference(inference_config)
        
        # Initialize noise
        x = torch.randn(n_samples, self.n_dims, device=self.device)
        
        # Generate samples
        with torch.no_grad():
            samples = self.lightning_module.predict_step({"input": x}, 0)
        
        return samples.cpu().numpy()
    
    def create_inference_gif(
        self,
        n_samples: int = 100,
        n_steps: int = 50,
        fps: int = 10,
        save_path: Optional[str] = None
    ):
        """Create a GIF showing the inference process."""
        print(f"Creating inference GIF with {n_steps} steps...")
        
        if save_path is None:
            save_path = self.output_dir / "inference.gif"
        
        if self.lightning_module is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Setup inference configuration with trajectory
        inference_config = self.setup_inference_config()
        inference_config.n_steps = n_steps
        inference_config.return_trajectory = True
        self.lightning_module.setup_inference(inference_config)
        
        # Initialize noise
        x = torch.randn(n_samples, self.n_dims, device=self.device)
        
        # Get training data for comparison
        train_data = torch.stack([self.train_dataset[i]["input"] for i in range(len(self.train_dataset))]).numpy()
        
        # Create figure and animation
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            
            # Plot training data
            ax.scatter(
                train_data[:, 0], train_data[:, 1],
                alpha=0.3, c='blue', s=20, label='Training Data'
            )
            
            # Get current state from trajectory
            with torch.no_grad():
                trajectory = self.lightning_module.predict_step({"input": x}, 0)
                if len(trajectory.shape) == 3:  # trajectory shape: [n_steps, batch_size, dims]
                    current_samples = trajectory[frame].cpu().numpy()
                else:  # single sample
                    current_samples = trajectory.cpu().numpy()
            
            # Plot current samples
            ax.scatter(
                current_samples[:, 0], current_samples[:, 1],
                alpha=0.7, c='red', s=30, label='Generated Samples'
            )
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_title(f'Diffusion Inference - Step {frame}/{n_steps}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Create animation
        anim = FuncAnimation(
            fig, animate, frames=n_steps, interval=1000//fps, repeat=False
        )
        
        # Save GIF
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Inference GIF saved to: {save_path}")
        
        plt.close()
    
    def visualize_results(self, n_samples: int = 1000):
        """Visualize training data and generated samples."""
        print("Visualizing results...")
        
        # Generate samples
        generated_samples = self.generate_samples(n_samples=n_samples)
        
        # Get training data
        train_data = torch.stack([self.train_dataset[i]["input"] for i in range(len(self.train_dataset))]).numpy()
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot training data
        ax1.scatter(train_data[:, 0], train_data[:, 1], alpha=0.6, s=20)
        ax1.set_title('Training Data (Swiss Roll)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        
        # Plot generated samples
        ax2.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, s=20, c='red')
        ax2.set_title('Generated Samples')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        # Plot both together
        ax3.scatter(train_data[:, 0], train_data[:, 1], alpha=0.4, s=20, label='Training Data')
        ax3.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, s=20, c='red', label='Generated Samples')
        ax3.set_title('Training Data vs Generated Samples')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "results_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results visualization saved to: {self.output_dir / 'results_comparison.png'}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        if self.lightning_module is None:
            # Initialize lightning module with training config
            training_config = self.setup_training_config()
            self.lightning_module = LightningDiffusion(
                model=self.denoiser,
                training_config=training_config
            )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.lightning_module.load_state_dict(state_dict)
        self.lightning_module.eval()
        print(f"Model loaded from {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Swiss Roll Diffusion Model Example")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-train", type=int, default=2**12, help="Number of training samples")
    parser.add_argument("--n-val", type=int, default=2**9, help="Number of validation samples")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension of the model")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--train-only", action="store_true", help="Only train the model")
    parser.add_argument("--inference-only", action="store_true", help="Only run inference (requires trained model)")
    parser.add_argument("--model-path", type=str, help="Path to trained model for inference")
    
    args = parser.parse_args()
    
    # Initialize the example
    example = SwissRollDiffusionExample(
        n_train=args.n_train,
        n_val=args.n_val,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    if args.inference_only:
        # Load trained model
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = example.output_dir / "model.pt"
        
        if not Path(model_path).exists():
            print(f"Error: Model file not found at {model_path}")
            print("Please train the model first or provide the correct model path.")
            return
        
        print(f"Loading model from {model_path}")
        example.load_model(str(model_path))
        
        # Run inference and create visualizations
        example.visualize_results()
        example.create_inference_gif()
        
    elif args.train_only:
        # Only train the model
        example.train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        
    else:
        # Full pipeline: train, inference, and visualization
        example.train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        example.visualize_results()
        example.create_inference_gif()


if __name__ == "__main__":
    main() 