#!/usr/bin/env python3
"""
Test script for UNet polygon colorization implementation.
This script verifies that all components work correctly.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from model import UNet, CombinedLoss
from dataset import SyntheticPolygonDataset
from utils import get_device, print_model_summary

def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    device = get_device()
    model = UNet(
        n_channels=3,
        n_classes=3,
        bilinear=True,
        color_embedding_dim=64,
        num_colors=10
    ).to(device)
    
    print_model_summary(model)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 128, 128).to(device)
    color_indices = torch.randint(0, 10, (batch_size,)).to(device)
    
    with torch.no_grad():
        output = model(input_tensor, color_indices)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Color indices: {color_indices}")
    
    assert output.shape == (batch_size, 3, 128, 128), f"Expected shape {(batch_size, 3, 128, 128)}, got {output.shape}"
    print("✓ Model forward pass successful!")

def test_dataset():
    """Test dataset creation and loading."""
    print("\nTesting dataset...")
    
    dataset = SyntheticPolygonDataset(num_samples=10, image_size=128)
    print(f"Dataset size: {len(dataset)}")
    
    # Test a few samples
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Output shape: {sample['output'].shape}")
        print(f"  Color index: {sample['color_idx']}")
        print(f"  Color name: {sample['color_name']}")
        
        assert sample['input'].shape == (3, 128, 128), f"Expected input shape (3, 128, 128), got {sample['input'].shape}"
        assert sample['output'].shape == (3, 128, 128), f"Expected output shape (3, 128, 128), got {sample['output'].shape}"
    
    print("✓ Dataset loading successful!")

def test_loss_function():
    """Test loss function."""
    print("\nTesting loss function...")
    
    device = get_device()
    criterion = CombinedLoss(l1_weight=0.5, ssim_weight=0.5)
    
    # Create dummy predictions and targets
    batch_size = 4
    predictions = torch.randn(batch_size, 3, 128, 128).to(device)
    targets = torch.randn(batch_size, 3, 128, 128).to(device)
    
    loss = criterion(predictions, targets)
    print(f"Loss value: {loss.item():.4f}")
    
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Loss function working!")

def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    device = get_device()
    
    # Create model, loss, and optimizer
    model = UNet(
        n_channels=3,
        n_classes=3,
        bilinear=True,
        color_embedding_dim=64,
        num_colors=10
    ).to(device)
    
    criterion = CombinedLoss(l1_weight=0.5, ssim_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy data
    batch_size = 2
    inputs = torch.randn(batch_size, 3, 128, 128).to(device)
    outputs = torch.randn(batch_size, 3, 128, 128).to(device)
    color_indices = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    predictions = model(inputs, color_indices)
    loss = criterion(predictions, outputs)
    
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Training step successful!")

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        from utils import load_config
        
        # Create a test config
        test_config = {
            'model': {
                'n_channels': 3,
                'n_classes': 3,
                'bilinear': True,
                'color_embedding_dim': 64,
                'image_size': 128
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 1e-4,
                'epochs': 100
            }
        }
        
        # Save and load config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        loaded_config = load_config(config_path)
        os.unlink(config_path)  # Clean up
        
        assert loaded_config['model']['n_channels'] == 3
        assert loaded_config['training']['batch_size'] == 16
        
        print("✓ Configuration loading successful!")
        
    except ImportError:
        print("⚠ PyYAML not available, skipping config test")

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing UNet Polygon Colorization Implementation")
    print("=" * 50)
    
    try:
        test_model_creation()
        test_dataset()
        test_loss_function()
        test_training_step()
        test_config_loading()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! Implementation is ready.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 