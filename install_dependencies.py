#!/usr/bin/env python3
"""
Install dependencies for the UNet polygon colorization project.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all required dependencies."""
    print("Installing dependencies for UNet Polygon Colorization...")
    
    # Core dependencies
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "matplotlib>=3.5.0",
        "wandb>=0.15.0",
        "jupyter>=1.0.0",
        "opencv-python>=4.5.0",
        "albumentations>=1.3.0",
        "tqdm>=4.64.0",
        "scikit-image>=0.19.0",
        "PyYAML>=6.0"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstallation complete: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("✓ All dependencies installed successfully!")
        print("\nYou can now run:")
        print("  python test_implementation.py")
        print("  python src/train.py --config configs/config.yaml")
    else:
        print("⚠ Some packages failed to install. You may need to install them manually.")
        print("Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 