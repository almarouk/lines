from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor
    from collections.abc import Callable

from torch.nn import Module, L1Loss, MSELoss, Identity
import torch

_map_loss: dict[str, Callable[[float], float]] = {
    'l1': L1Loss,
    'l2': MSELoss,
}

def get_loss_fn(
        loss_type: str,
        reduction: str = "mean",
        weight_valid : float = None,
        **kwargs
    ) -> Module:
    assert weight_valid <= 1 and weight_valid >=0
    assert reduction in ['mean', 'sum', 'none']
    assert loss_type in _map_loss
    if weight_valid is None:
        return _map_loss[loss_type](reduction)
    else:
        return WeightedLoss(loss_type, weight_valid, reduction)

class WeightedLoss(Module):
    def __init__(self, loss_type: str, weight_valid: float, reduction: str = 'mean') -> None:
        super().__init__()
        self.loss_type = loss_type
        self.weight_valid = weight_valid
        self.reduction = reduction

        self._init()
            
    def _init(self) -> None:
        self.loss_fn = _map_loss[self.loss_type](reduction="none")
        if self.reduction == "none":
            self.reduce_fn = Identity()
        elif self.reduction == "sum":
            self.reduce_fn = torch.sum
        elif self.reduction == "mean":
            self.reduce_fn = torch.mean

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            mask = (target < 1).float()
            weights = self.weight_valid * mask / mask.sum((-1, -2), True)
            mask = 1 - mask
            weights += (1 - self.weight_valid) * mask / mask.sum((-1, -2), True)
        loss = self.loss_fn(input, target) * weights
        return self.reduce_fn(loss)