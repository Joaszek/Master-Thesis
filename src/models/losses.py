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
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        # Clamp to prevent exp(-large) = 0 → (1-0)^gamma * large = large → NaN
        ce_loss = torch.clamp(ce_loss, max=50.0)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
