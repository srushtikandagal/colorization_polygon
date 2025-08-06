import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import yaml
from typing import Dict, Any, Optional
import wandb


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, 
                   save_path: str, config: Dict[str, Any]):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, save_path)


def load_checkpoint(model: nn.Module, optimizer, checkpoint_path: str):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def visualize_batch(inputs, outputs, predictions, color_names, save_path: Optional[str] = None):
    """Visualize a batch of inputs, outputs, and predictions."""
    batch_size = inputs.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Input
        input_img = inputs[i].cpu().numpy().transpose(1, 2, 0)
        input_img = np.clip(input_img, 0, 1)
        axes[i, 0].imshow(input_img)
        axes[i, 0].set_title(f'Input Polygon')
        axes[i, 0].axis('off')
        
        # Target
        target_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
        target_img = np.clip(target_img, 0, 1)
        axes[i, 1].imshow(target_img)
        axes[i, 1].set_title(f'Target ({color_names[i]})')
        axes[i, 1].axis('off')
        
        # Prediction
        pred_img = predictions[i].cpu().numpy().transpose(1, 2, 0)
        pred_img = np.clip(pred_img, 0, 1)
        axes[i, 2].imshow(pred_img)
        axes[i, 2].set_title(f'Prediction ({color_names[i]})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def log_images_to_wandb(inputs, outputs, predictions, color_names, step: int):
    """Log images to wandb."""
    batch_size = inputs.shape[0]
    
    for i in range(min(batch_size, 4)):  # Log first 4 images
        input_img = inputs[i].cpu().numpy().transpose(1, 2, 0)
        target_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
        pred_img = predictions[i].cpu().numpy().transpose(1, 2, 0)
        
        # Convert to PIL images
        input_pil = Image.fromarray((input_img * 255).astype(np.uint8))
        target_pil = Image.fromarray((target_img * 255).astype(np.uint8))
        pred_pil = Image.fromarray((pred_img * 255).astype(np.uint8))
        
        wandb.log({
            f"sample_{i}_input": wandb.Image(input_pil, caption=f"Input Polygon"),
            f"sample_{i}_target": wandb.Image(target_pil, caption=f"Target ({color_names[i]})"),
            f"sample_{i}_prediction": wandb.Image(pred_pil, caption=f"Prediction ({color_names[i]})")
        }, step=step)


def calculate_metrics(predictions, targets):
    """Calculate various metrics."""
    # L1 Loss
    l1_loss = torch.mean(torch.abs(predictions - targets))
    
    # MSE Loss
    mse_loss = torch.mean((predictions - targets) ** 2)
    
    # PSNR
    mse = torch.mean((predictions - targets) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return {
        'l1_loss': l1_loss.item(),
        'mse_loss': mse_loss.item(),
        'psnr': psnr.item()
    }


def setup_wandb(config: Dict[str, Any], project_name: str = "polygon-colorization"):
    """Setup wandb logging."""
    wandb.init(
        project=project_name,
        config=config,
        name=f"unet_colorization_{config['model']['image_size']}x{config['model']['image_size']}",
        tags=["unet", "colorization", "polygon"]
    )


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory."""
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if image_tensor.device != mean.device:
        mean = mean.to(image_tensor.device)
        std = std.to(image_tensor.device)
    
    return image_tensor * std + mean


def normalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize image tensor."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if image_tensor.device != mean.device:
        mean = mean.to(image_tensor.device)
        std = std.to(image_tensor.device)
    
    return (image_tensor - mean) / std


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """Print model summary."""
    total_params = count_parameters(model)
    print(f"Model Summary:")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Print layer information
    print("\nLayer Information:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name}: {params:,} parameters")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 