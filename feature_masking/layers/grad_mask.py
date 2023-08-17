# Description: This file contains the function to mask out the gradients of the feature map
import torch
from torch import Tensor


def gradient_mask(feat_map: Tensor, mask: Tensor, train: bool) -> Tensor:
    # Re-initialize the gradient
    feature_map_grad = feat_map.clone().requires_grad_(train)
    backward_hook = None
    if train:
        # Mask out the gradients of the feature map
        backward_hook = feature_map_grad.register_hook(lambda grad: grad.mul_(mask))
    return feature_map_grad, backward_hook
