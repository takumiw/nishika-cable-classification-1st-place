import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothingCrossEntropy(nn.Module):
    """
    label_smoothing = 0.02 makes your binary targets 0.01 and 0.99
    label_smoothing is a ratio of smoothing, i.e. label_smoothing = 1.0 push all targets
    all the way to 0.5 (which is not particularly useful ;)
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean", weight=None) -> None:
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss: Tensor) -> Tensor:
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x: Tensor, y: Tensor) -> Tensor:
        return self.epsilon * x + (1 - self.epsilon) * y

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        if self.training:
            n = preds.size(-1)
            log_preds = F.log_softmax(preds, dim=-1)
            loss = self.reduce_loss(-log_preds.sum(dim=-1))
            nll = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weight)
            return self.linear_combination(loss / n, nll)
        else:
            return F.cross_entropy(preds, target, weight=self.weight)
