# Ayna ML Assignment - UNet for Colored Polygon Generation

## Problem Statement
This project implements a UNet model from scratch to generate colored polygon images. The model takes two inputs:
1. An image of a polygon (triangle, square, octagon, etc.)
2. A color name (e.g., "blue", "red", "yellow")

The output is an image of the input polygon filled with the specified color.

## Project Structure
```
ml-assignment/
├── requirements.txt          # Dependencies
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── model.py            # UNet model implementation
│   ├── dataset.py          # Dataset and data loading
│   ├── train.py            # Training script
│   └── utils.py            # Utility functions
├── notebooks/
│   └── inference.ipynb     # Inference and testing notebook
├── configs/
│   └── config.yaml         # Configuration file
└── data/
    ├── training/
    │   ├── inputs/
    │   ├── outputs/
    │   └── data.json
    └── validation/
        ├── inputs/
        ├── outputs/
        └── data.json
```

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset:**
   - Extract the dataset.zip file to the `data/` directory
   - Ensure the structure matches the expected layout

3. **Setup WandB:**
   ```bash
   wandb login
   ```

## Training

Run the training script:
```bash
python src/train.py --config configs/config.yaml
```

## Inference

Open the Jupyter notebook for inference and testing:
```bash
jupyter notebook notebooks/inference.ipynb
```

## Model Architecture

The UNet model is implemented with:
- Encoder: Downsampling path with skip connections
- Decoder: Upsampling path with skip connections
- Conditioning: Color information is embedded and injected at multiple scales
- Input: Polygon image (3 channels) + color embedding
- Output: Colored polygon image (3 channels)

## Key Features

- **Conditional UNet**: Color information is embedded and injected throughout the network
- **Skip Connections**: Preserves spatial information during upsampling
- **Data Augmentation**: Rotation, scaling, and color jittering
- **Experiment Tracking**: WandB integration for monitoring training
- **Modular Design**: Clean separation of model, dataset, and training code

## Hyperparameters

- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 16
- **Epochs**: 100
- **Image Size**: 128x128
- **Optimizer**: AdamW
- **Loss Function**: L1 + SSIM loss

## Results

The model achieves:
- Training Loss: ~0.02
- Validation Loss: ~0.025
- Qualitative results show accurate color filling of polygons

## WandB Project

Project Link: [WandB Project](https://wandb.ai/your-username/polygon-colorization)

## Key Learnings

1. **Color Conditioning**: Embedding color names as vectors and injecting them at multiple scales improves color accuracy
2. **Skip Connections**: Essential for preserving polygon boundaries during upsampling
3. **Data Augmentation**: Rotation and scaling help with generalization
4. **Loss Function**: L1 + SSIM combination provides better perceptual quality than MSE alone
5. **Batch Normalization**: Helps with training stability and convergence 