import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ColorEmbedding(nn.Module):
    """Embed color names into a vector representation."""
    
    def __init__(self, num_colors, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_colors, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, color_indices):
        # color_indices: (batch_size,)
        embedded = self.embedding(color_indices)  # (batch_size, embedding_dim)
        projected = self.projection(embedded)     # (batch_size, embedding_dim)
        return projected


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ColorInjection(nn.Module):
    """Inject color information into feature maps."""
    
    def __init__(self, feature_channels, color_dim):
        super().__init__()
        self.color_projection = nn.Linear(color_dim, feature_channels)
        self.conv = nn.Conv2d(feature_channels * 2, feature_channels, 1)
        
    def forward(self, features, color_embedding):
        # features: (batch_size, channels, height, width)
        # color_embedding: (batch_size, color_dim)
        
        # Project color to feature space
        color_features = self.color_projection(color_embedding)  # (batch_size, channels)
        
        # Expand color features to spatial dimensions
        batch_size, channels, height, width = features.shape
        color_spatial = color_features.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, channels, height, width
        )
        
        # Concatenate and project
        combined = torch.cat([features, color_spatial], dim=1)
        return self.conv(combined)


class UNet(nn.Module):
    """UNet model with color conditioning."""
    
    def __init__(self, n_channels=3, n_classes=3, bilinear=False, 
                 color_embedding_dim=64, num_colors=10):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Color embedding
        self.color_embedding = ColorEmbedding(num_colors, color_embedding_dim)
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Color injection layers
        self.color_injection1 = ColorInjection(64, color_embedding_dim)
        self.color_injection2 = ColorInjection(128, color_embedding_dim)
        self.color_injection3 = ColorInjection(256, color_embedding_dim)
        self.color_injection4 = ColorInjection(512, color_embedding_dim)
        self.color_injection5 = ColorInjection(1024 // factor, color_embedding_dim)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Color injection in decoder
        self.color_injection_up1 = ColorInjection(512 // factor, color_embedding_dim)
        self.color_injection_up2 = ColorInjection(256 // factor, color_embedding_dim)
        self.color_injection_up3 = ColorInjection(128 // factor, color_embedding_dim)
        self.color_injection_up4 = ColorInjection(64, color_embedding_dim)

    def forward(self, x, color_indices):
        # x: (batch_size, channels, height, width)
        # color_indices: (batch_size,)
        
        # Get color embedding
        color_embedding = self.color_embedding(color_indices)  # (batch_size, color_dim)
        
        # Encoder path
        x1 = self.inc(x)
        x1 = self.color_injection1(x1, color_embedding)
        
        x2 = self.down1(x1)
        x2 = self.color_injection2(x2, color_embedding)
        
        x3 = self.down2(x2)
        x3 = self.color_injection3(x3, color_embedding)
        
        x4 = self.down3(x3)
        x4 = self.color_injection4(x4, color_embedding)
        
        x5 = self.down4(x4)
        x5 = self.color_injection5(x5, color_embedding)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.color_injection_up1(x, color_embedding)
        
        x = self.up2(x, x3)
        x = self.color_injection_up2(x, color_embedding)
        
        x = self.up3(x, x2)
        x = self.color_injection_up3(x, color_embedding)
        
        x = self.up4(x, x1)
        x = self.color_injection_up4(x, color_embedding)
        
        logits = self.outc(x)
        return logits


class SSIMLoss(nn.Module):
    """SSIM Loss for better perceptual quality."""
    
    def __init__(self, window_size=11, size_average=True, channel=3):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class CombinedLoss(nn.Module):
    """Combined L1 and SSIM loss."""
    
    def __init__(self, l1_weight=0.5, ssim_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim 