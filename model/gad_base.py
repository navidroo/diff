from random import randrange
from re import I

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

INPUT_DIM = 4
FEATURE_DIM = 64

class SelfAttentionBlock(nn.Module):
    """Transformer-style self-attention block for capturing long-range dependencies."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm([dim])
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm([dim])
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape to sequence for attention
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Layer norm expects [B, *, dim]
        x_norm = self.norm1(x_flat)
        
        # Self-attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # MLP block
        x_norm = self.norm2(x_flat)
        x_flat = x_flat + self.mlp(x_norm)
        
        # Reshape back to spatial
        x = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class TransformerDiffusion(nn.Module):
    """Enhances the diffusion process with transformer blocks for long-range dependencies."""
    def __init__(self, feature_dim, num_blocks=2, num_heads=8):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            SelfAttentionBlock(feature_dim, num_heads=num_heads)
            for _ in range(num_blocks)
        ])
        
    def forward(self, features):
        transformed = features
        for block in self.transformer_blocks:
            transformed = block(transformed)
        return transformed

class GADBase(nn.Module):
    
    def __init__(
            self, feature_extractor='Unet',
            Npre=8000, Ntrain=1024,
            use_transformer=True,
            transformer_blocks=2,
            transformer_heads=8
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        self.use_transformer = use_transformer
 
        if feature_extractor=='none': 
            # RGB verion of DADA does not need a deep feature extractor
            self.feature_extractor = None
            self.Ntrain = 0
            self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            # Learned verion of DADA
            self.feature_extractor =  torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bicubic'),
                smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=INPUT_DIM),
                torch.nn.AvgPool2d(kernel_size=2, stride=2) ).cuda()
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))
            
            # Add transformer module for high-scaling factors
            if use_transformer:
                self.transformer = TransformerDiffusion(FEATURE_DIM, 
                                                       num_blocks=transformer_blocks,
                                                       num_heads=transformer_heads)
        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')
             

    def forward(self, sample, train=False, deps=0.1):
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        # assert that all values are positive, otherwise shift depth map to positives
        if source.min()<=deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train)

        # revert the shift
        if shifted:
            y_pred -= deps

        # return {'y_pred': y_pred} | aux
        return {**{'y_pred': y_pred}, **aux}


    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):

        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape

        # Define Downsampling operations that depend on the input size
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Deep Learning version or RGB version to calucalte the coefficients
        if self.feature_extractor is None: 
            guide_feats = torch.cat([guide, img], 1) 
        else:
            guide_feats = self.feature_extractor(torch.cat([guide, img-img.mean((1,2,3), keepdim=True) ], 1))
            
            # Apply transformer for long-range dependencies if enabled
            if self.use_transformer:
                guide_feats = self.transformer(guide_feats)
        
        # Convert the features to coefficients with the Perona-Malik edge-detection function
        cv, ch = c(guide_feats, K=K)

        # Iterations without gradient
        if self.Npre>0: 
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                for t in range(Npre):                     
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)

        # Iterations with gradient
        if self.Ntrain>0: 
            for t in range(self.Ntrain): 
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)

        return img, {"cv": cv, "ch": ch}


# @torch.jit.script
def c(I, K: float=0.03):
    # apply function to both dimensions
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
    return cv, ch

# @torch.jit.script
def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))

@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
    # Anisotropic Diffusion implmentation, Eq. (1) in paper.

    # calculate diffusion update as increments
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th 
    
    return I

def adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8):
    # Implementation of the adjustment step. Eq (3) in paper.

    # Iss = subsample img
    img_ss = downsample(img)

    # Rss = source / Iss
    ratio_ss = source / (img_ss + eps)
    ratio_ss[mask_inv] = 1

    # R = NN upsample r
    ratio = upsample(ratio_ss)

    # ratio = torch.sqrt(ratio)
    # img = img * R
    return img * ratio 
