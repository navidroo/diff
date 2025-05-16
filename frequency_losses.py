import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MagnitudeFrequencyLoss(nn.Module):
    """
    Magnitude-only Frequency Loss for spectral-domain supervision in depth super-resolution.
    Removes phase components that can cause instability.
    """
    def __init__(self, alpha=1.0, use_focal_weight=True, focal_lambda=0.5, eps=1e-8, norm="ortho"):
        """
        Initialize the Magnitude-only Frequency Loss

        Args:
            alpha: Weight for the magnitude loss
            use_focal_weight: Whether to use focal weighting
            focal_lambda: Controls the non-linear scaling of frequency errors
            eps: Small constant for numerical stability
            norm: Normalization method for FFT ("ortho" recommended for stability)
        """
        super().__init__()
        self.alpha = alpha
        self.use_focal_weight = use_focal_weight
        self.focal_lambda = focal_lambda
        self.eps = eps
        self.norm = norm
        
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted depth map (B, 1, H, W)
            target: Ground truth depth map (B, 1, H, W)
            mask: Optional mask to ignore certain regions (B, 1, H, W)
        """
        if mask is not None:
            # Apply mask if provided
            valid_mask = mask == 1.0
            if not valid_mask.any():
                return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
                
            # Only compute loss on valid regions
            pred = pred * valid_mask
            target = target * valid_mask
        
        # Get tensor dimensions
        batch_size, channels, height, width = pred.shape
        
        # Compute full 2D FFT with normalization (returns complex tensors)
        pred_fft = torch.fft.fft2(pred, norm=self.norm)
        target_fft = torch.fft.fft2(target, norm=self.norm)
        
        # Shift zero frequency to center for better frequency weighting
        pred_fft = torch.fft.fftshift(pred_fft)
        target_fft = torch.fft.fftshift(target_fft)
        
        # Extract magnitude
        pred_mag = torch.abs(pred_fft) + self.eps
        target_mag = torch.abs(target_fft) + self.eps
        
        # Magnitude error
        mag_diff = torch.abs(pred_mag - target_mag)
        
        # Create frequency weight matrix - distance from center (zero frequency)
        # Create meshgrid in normalized [-1, 1] coordinate system
        y_grid = torch.linspace(-1, 1, height, device=pred.device)
        x_grid = torch.linspace(-1, 1, width, device=pred.device)
        y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Calculate distance from center
        freq_grid = torch.sqrt(y_grid**2 + x_grid**2)
        
        # Unsqueeze to add batch and channel dimensions
        freq_grid = freq_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)
        
        # Apply frequency emphasis
        weighted_mag_diff = mag_diff * freq_grid
        
        # Focal weighting if enabled
        if self.use_focal_weight:
            # Normalize relative error for stability
            rel_error = mag_diff / (target_mag.mean() + self.eps)
            # Limit maximum focal weight for stability
            focal_weights = 1 - torch.exp(-self.focal_lambda * rel_error)
            focal_weights = torch.clamp(focal_weights, 0.1, 2.0)
            weighted_mag_diff = weighted_mag_diff * focal_weights
        
        # Compute loss
        mag_loss = weighted_mag_diff.mean()
        
        return self.alpha * mag_loss, mag_loss
        

class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss for spectral-domain supervision in depth super-resolution.
    With added phase consistency component.
    """
    def __init__(self, alpha=1.0, beta=1.0, focal_lambda=0.5, eps=1e-8):
        """
        Initialize the Focal Frequency Loss

        Args:
            alpha: Weight for the magnitude loss
            beta: Weight for the phase loss
            focal_lambda: Controls the non-linear scaling of frequency errors
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.alpha = alpha        # Weight for magnitude loss
        self.beta = beta          # Weight for phase loss
        self.focal_lambda = focal_lambda  # Controls focal weighting strength
        self.eps = eps            # Small constant to prevent division by zero
        
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted depth map (B, 1, H, W)
            target: Ground truth depth map (B, 1, H, W)
            mask: Optional mask to ignore certain regions (B, 1, H, W)
        """
        if mask is not None:
            # Apply mask if provided
            valid_mask = mask == 1.0
            if not valid_mask.any():
                return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
                
            # Only compute loss on valid regions
            pred = pred * valid_mask
            target = target * valid_mask
        
        # Get tensor dimensions
        batch_size, channels, height, width = pred.shape
        
        # Compute full 2D FFT (returns complex tensors)
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Shift zero frequency to center for better frequency weighting
        pred_fft = torch.fft.fftshift(pred_fft)
        target_fft = torch.fft.fftshift(target_fft)
        
        # Extract magnitude and phase
        pred_mag = torch.abs(pred_fft) + self.eps
        target_mag = torch.abs(target_fft) + self.eps
        
        # Extract phase using torch.angle
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Magnitude error
        mag_diff = torch.abs(pred_mag - target_mag)
        
        # Phase error - compute circular difference to handle 2π wraparound
        phase_diff = torch.abs(torch.atan2(
            torch.sin(pred_phase - target_phase), 
            torch.cos(pred_phase - target_phase)
        ))
        
        # Create frequency weight matrix - distance from center (zero frequency)
        # Create meshgrid in normalized [-1, 1] coordinate system
        y_grid = torch.linspace(-1, 1, height, device=pred.device)
        x_grid = torch.linspace(-1, 1, width, device=pred.device)
        y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Calculate distance from center
        freq_grid = torch.sqrt(y_grid**2 + x_grid**2)
        
        # Unsqueeze to add batch and channel dimensions
        freq_grid = freq_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)
        
        # Focal weighting using non-linear scaling (sigmoid-based)
        # This applies greater weights to harder-to-learn frequencies
        rel_error = mag_diff / (target_mag + self.eps)
        focal_weights = 1 - torch.exp(-self.focal_lambda * rel_error)  # Error-based focal weighting

        # Apply frequency emphasis and focal weighting to magnitude loss
        weighted_mag_diff = mag_diff * freq_grid * focal_weights
        
        # Weight phase loss by frequency importance and signal strength
        phase_weights = freq_grid * target_mag
        weighted_phase_diff = phase_diff * phase_weights / (target_mag.max() + self.eps)
        
        # Compute separate loss components
        mag_loss = weighted_mag_diff.mean()
        phase_loss = weighted_phase_diff.mean()
        
        # Combine weighted loss components
        total_loss = self.alpha * mag_loss + self.beta * phase_loss
        
        return total_loss, mag_loss, phase_loss
        

class LogFocalFrequencyLoss(nn.Module):
    """
    Alternative implementation using logarithmic-based focal weighting
    """
    def __init__(self, alpha=1.0, beta=1.0, focal_gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha        # Weight for magnitude loss
        self.beta = beta          # Weight for phase loss
        self.focal_gamma = focal_gamma  # Controls focal weighting strength (higher = more emphasis on hard frequencies)
        self.eps = eps
        
    def forward(self, pred, target, mask=None):
        if mask is not None:
            valid_mask = mask == 1.0
            if not valid_mask.any():
                return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
            pred = pred * valid_mask
            target = target * valid_mask
            
        # Get tensor dimensions
        b, c, h, w = pred.shape
            
        # Full FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Shift to put zero-frequency in center
        pred_fft = torch.fft.fftshift(pred_fft)
        target_fft = torch.fft.fftshift(target_fft)
        
        # Get magnitude and phase
        pred_mag = torch.abs(pred_fft) + self.eps
        target_mag = torch.abs(target_fft) + self.eps
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Compute errors
        mag_diff = torch.abs(pred_mag - target_mag)
        phase_diff = torch.abs(torch.atan2(torch.sin(pred_phase - target_phase), torch.cos(pred_phase - target_phase)))
        
        # Create frequency distance weights
        # Create meshgrid in normalized [-1, 1] coordinate system
        y_grid = torch.linspace(-1, 1, h, device=pred.device)
        x_grid = torch.linspace(-1, 1, w, device=pred.device)
        y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Calculate distance from center
        freq_grid = torch.sqrt(y_grid**2 + x_grid**2)
        
        # Expand grid to match batch and channel dimensions
        freq_grid = freq_grid.unsqueeze(0).unsqueeze(0).expand(b, c, h, w)
        
        # Logarithmic-based focal weighting - puts more emphasis on higher error regions
        rel_error = mag_diff / (target_mag + self.eps)
        log_term = torch.log1p(rel_error * 10) / math.log(11)  # Normalized log scaling
        focal_weights = log_term ** self.focal_gamma
        
        # Apply weights
        weighted_mag_diff = mag_diff * freq_grid * focal_weights
        weighted_phase_diff = phase_diff * freq_grid * target_mag / (target_mag.max() + self.eps)
        
        # Final losses
        mag_loss = weighted_mag_diff.mean()
        phase_loss = weighted_phase_diff.mean()
        
        # Combined loss
        total_loss = self.alpha * mag_loss + self.beta * phase_loss
        
        return total_loss, mag_loss, phase_loss


def get_frequency_loss(output, sample, alpha=1.0, beta=1.0, phase_weight=0.5, 
                       use_log_focal=False, focal_gamma=2.0, focal_lambda=0.5,
                       magnitude_only=False, use_focal_weight=True, norm="ortho"):
    """
    Wrapper function to compute the frequency loss

    Args:
        output: Model output dictionary containing 'y_pred'
        sample: Dictionary with ground truth and masks
        alpha: Weight for magnitude component
        beta: Weight for phase component 
        phase_weight: Legacy parameter, ignored if alpha and beta are specified
        use_log_focal: Whether to use the logarithmic focal weighting
        focal_gamma: Controls the strength of focal weighting (for log-based)
        focal_lambda: Controls the strength of sigmoid-based focal weighting
        magnitude_only: If True, only use magnitude component (no phase)
        use_focal_weight: Whether to use focal weighting at all
        norm: Normalization method for FFT
    """
    y_pred = output['y_pred']
    y, mask_hr = sample['y'], sample['mask_hr']
    
    # Use appropriate version of the frequency loss
    if magnitude_only:
        freq_loss_fn = MagnitudeFrequencyLoss(
            alpha=alpha, 
            use_focal_weight=use_focal_weight,
            focal_lambda=focal_lambda,
            norm=norm
        )
        freq_loss, mag_loss = freq_loss_fn(y_pred, y, mask_hr)
        phase_loss = torch.tensor(0.0, device=y_pred.device)
    elif use_log_focal:
        freq_loss_fn = LogFocalFrequencyLoss(alpha=alpha, beta=beta, focal_gamma=focal_gamma)
        freq_loss, mag_loss, phase_loss = freq_loss_fn(y_pred, y, mask_hr)
    else:
        # For compatibility with phase_weight parameter
        if phase_weight < 1.0 and (alpha == 1.0 and beta == 1.0):  # Only use phase_weight if alpha and beta are default
            alpha_scaled = 1.0 - phase_weight
            beta_scaled = phase_weight
        else:
            alpha_scaled = alpha
            beta_scaled = beta
            
        freq_loss_fn = FocalFrequencyLoss(alpha=alpha_scaled, beta=beta_scaled, focal_lambda=focal_lambda)
        freq_loss, mag_loss, phase_loss = freq_loss_fn(y_pred, y, mask_hr)
    
    # Also compute standard losses for reference/comparison
    mse_loss = F.mse_loss(y_pred[mask_hr == 1.], y[mask_hr == 1.])
    l1_loss = F.l1_loss(y_pred[mask_hr == 1.], y[mask_hr == 1.])
    
    # Use frequency loss for optimization
    loss = freq_loss
    
    return loss, {
        'freq_loss': freq_loss.detach().item(),
        'mag_loss': mag_loss.detach().item(), 
        'phase_loss': phase_loss.detach().item() if not magnitude_only else 0.0,
        'l1_loss': l1_loss.detach().item(),
        'mse_loss': mse_loss.detach().item(),
        'optimization_loss': loss.detach().item(),
        'raw_loss': loss  # Add raw loss to ensure tensor with gradients is accessible
    } 