import torch
from torch.nn import BCEWithLogitsLoss


class MaskedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true):
        y_pred, mask = y_pred['y_pred'], y_pred['mask']
        loss = self.bce(y_pred, y_true)
        loss *= mask.unsqueeze(-1)
        loss = loss.mean()
        return loss
