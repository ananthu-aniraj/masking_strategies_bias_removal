import torch
from torch import Tensor
from .grad_mask import gradient_mask


# Baseline model, a modified ConvNext with feature masking
class FeatureMaskingNet(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, seg_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.seg_model = seg_model
        self.backward_hook = None

    def forward_feat_ddp(self, x: Tensor, train: bool = False) -> Tensor:
        # Forward pass through the convolutional backbone
        feature_map = self.base_model.module.forward_features(x)

        with torch.no_grad():
            batch_img_metas = [
                                  dict(
                                      ori_shape=x.shape[2:],
                                      img_shape=x.shape[2:],
                                      pad_shape=x.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * x.shape[0]
            outputs_seg = torch.argmax(self.seg_model.module.inference(x, batch_img_metas), dim=1,
                                       keepdim=True)
            # Down-sample the segmentation mask to the feature map size
            downsampled_mask: Tensor = torch.nn.functional.interpolate(outputs_seg.float(), size=(
                feature_map.shape[-2], feature_map.shape[-1]), mode='nearest-exact')
            # Multiply the feature map with the segmentation mask
            feature_map.mul_(downsampled_mask)
        feature_map_grad, self.backward_hook = gradient_mask(feature_map, downsampled_mask, train)
        # Forward pass through the head
        out = self.base_model.module.forward_head(feature_map_grad)
        return out

    def forward_feat(self, x: Tensor, train: bool = False) -> Tensor:
        # Forward pass through the convolutional backbone
        feature_map = self.base_model.forward_features(x)

        with torch.no_grad():
            batch_img_metas = [
                                  dict(
                                      ori_shape=x.shape[2:],
                                      img_shape=x.shape[2:],
                                      pad_shape=x.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * x.shape[0]
            outputs_seg = torch.argmax(self.seg_model.inference(x, batch_img_metas), dim=1,
                                       keepdim=True)
            # Down-sample the segmentation mask to the feature map size
            downsampled_mask: Tensor = torch.nn.functional.interpolate(outputs_seg.float(), size=(
                feature_map.shape[-2], feature_map.shape[-1]), mode='nearest-exact')
            # Multiply the feature map with the segmentation mask
            feature_map.mul_(downsampled_mask)
        # Re-initialize the gradient
        feature_map_grad, self.backward_hook = gradient_mask(feature_map, downsampled_mask, train)
        # Forward pass through the head
        out = self.base_model.forward_head(feature_map_grad)
        return out
