#!/usr/bin/env python3
"""
Simple Swiss Roll Diffusion Example Runner

This script provides a simple way to run the Swiss roll diffusion example.
It will train a model and generate visualizations including a GIF.
"""

import sys
from pathlib import Path

# Add the diffusion package to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from swiss_roll_example import SwissRollDiffusionExample


def main():
    """Run the complete Swiss roll diffusion example."""
    print("🚀 Starting Swiss Roll Diffusion Example")
    print("=" * 50)
    
    # Initialize the example
    example = SwissRollDiffusionExample(
        n_train=2**12,      # 4096 training samples
        n_val=2**9,         # 512 validation samples
        hidden_dim=256,     # Hidden dimension
        num_layers=2,       # Number of layers
    )
    
    print(f"📊 Dataset created:")
    print(f"   - Training samples: {len(example.train_dataset)}")
    print(f"   - Validation samples: {len(example.val_dataset)}")
    print(f"   - Input dimensions: {example.n_dims}")
    print(f"   - Device: {example.device}")
    print()
    
    # Train the model
    print("🏋️  Training the diffusion model...")
    example.train(
        epochs=30,          # Number of epochs
        batch_size=128,     # Batch size
        lr=1e-4            # Learning rate
    )
    print("✅ Training completed!")
    print()
    
    # Generate visualizations
    print("🎨 Generating visualizations...")
    example.visualize_results(n_samples=1000)
    print("✅ Visualizations completed!")
    print()
    
    # Create inference GIF
    print("🎬 Creating inference GIF...")
    example.create_inference_gif(
        n_samples=100,      # Number of samples for GIF
        n_steps=50,         # Number of inference steps
        fps=10              # Frames per second
    )
    print("✅ GIF creation completed!")
    print()
    
    print("🎉 All done! Check the 'swiss_roll_output' directory for results:")
    print("   - model.pt: Trained model")
    print("   - results_comparison.png: Training vs generated samples comparison")
    print("   - inference.gif: Animation of the inference process")
    print("   - lightning_logs/: Training logs")


if __name__ == "__main__":
    main() 