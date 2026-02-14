import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss dla class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights (tensor[num_classes]) lub float
        gamma: Focusing parameter (typowo 2.0)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
        """
        # Clamp logits to prevent extreme values before cross_entropy
        inputs = torch.clamp(inputs, min=-30.0, max=30.0)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        ce_loss = torch.clamp(ce_loss, min=0.0, max=50.0)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        # Replace any NaN with zero to prevent propagation
        focal_loss = torch.nan_to_num(focal_loss, nan=0.0)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
