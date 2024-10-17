from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor
    from collections.abc import Callable

from torch.nn import Module, L1Loss, MSELoss

_map_loss: dict[str, tuple[type[Module], Callable[[float], float]]] = {
    'l1': (L1Loss, lambda x: abs(x)),
    'l2': (MSELoss, lambda x: x * x),
}

class Loss(Module):
    __constants__ = ['loss_fn', 'distance']
    loss_fn: Module
    distance: float

    def __init__(self, loss: str, distance: float = 1., reduction: str = 'mean') -> None:
        super().__init__()
        loss_type, distance_fn = _map_loss[loss]
        self.loss_fn = loss_type(reduction=reduction)
        self.distance = distance_fn(distance)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss_fn(input, target) / self.distance