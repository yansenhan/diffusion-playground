#!/usr/bin/env python3
"""
Test script for Swiss Roll Diffusion Example

This script tests the basic functionality without full training.
"""

import sys
from pathlib import Path

# Add the diffusion package to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import matplotlib.pyplot as plt
from swiss_roll_example import SwissRollDiffusionExample


def test_data_creation():
    """Test data creation and visualization."""
    print("Testing data creation...")
    
    example = SwissRollDiffusionExample(
        n_train=100,  # Small dataset for testing
        n_val=50,
        hidden_dim=64,  # Smaller model for testing
        num_layers=1
    )
    
    # Test dataset creation
    assert len(example.train_dataset) == 100
    assert len(example.val_dataset) == 50
    
    # Test data format
    sample = example.train_dataset[0]
    assert "input" in sample
    assert sample["input"].shape == (2,)
    
    print("✅ Data creation test passed!")
    return example


def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    example = SwissRollDiffusionExample(
        n_train=10,
        n_val=5,
        hidden_dim=64,
        num_layers=1
    )
    
    # Test model forward pass
    x = torch.randn(5, 2, device=example.device)
    t = torch.rand(5, device=example.device)
    
    with torch.no_grad():
        output = example.model(x, t)
        assert output.shape == (5, 2)
    
    print("✅ Model creation test passed!")
    return example


def test_denoiser_creation():
    """Test denoiser creation."""
    print("Testing denoiser creation...")
    
    example = SwissRollDiffusionExample(
        n_train=10,
        n_val=5,
        hidden_dim=64,
        num_layers=1
    )
    
    # Test denoiser forward pass
    x = torch.randn(5, 2, device=example.device)
    sigma = torch.rand(5, device=example.device)
    
    with torch.no_grad():
        output = example.denoiser(x, sigma)
        assert output.shape == (5, 2)
    
    print("✅ Denoiser creation test passed!")
    return example


def test_visualization():
    """Test visualization functions."""
    print("Testing visualization...")
    
    example = SwissRollDiffusionExample(
        n_train=100,
        n_val=50,
        hidden_dim=64,
        num_layers=1
    )
    
    # Create a simple visualization
    train_data = torch.stack([example.train_dataset[i]["input"] for i in range(len(example.train_dataset))]).numpy()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(train_data[:, 0], train_data[:, 1], alpha=0.6, s=20)
    plt.title('Swiss Roll Training Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualization test passed!")
    print("   - Check 'test_visualization.png' for the plot")


def main():
    """Run all tests."""
    print("🧪 Running Swiss Roll Diffusion Example Tests")
    print("=" * 50)
    
    try:
        test_data_creation()
        test_model_creation()
        test_denoiser_creation()
        test_visualization()
        
        print("\n🎉 All tests passed!")
        print("The Swiss Roll Diffusion Example is ready to use.")
        print("\nTo run the full example:")
        print("python run_swiss_roll_example.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 