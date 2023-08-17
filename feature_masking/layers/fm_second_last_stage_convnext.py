import torch
from torch import Tensor
from .grad_mask import gradient_mask


# Baseline model, a modified ConvNext with feature masking
class FeatureMaskingNetSecondLastStage(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, seg_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.seg_model = seg_model
        self.backward_hook = None
        self.num_stages = len(self.base_model.stages)

    def forward_feat_ddp(self, x: Tensor, train: bool = False) -> Tensor:
        # Forward pass through the convolutional stem
        feature_map = self.base_model.module.stem(x)
        # Forward pass through the convolutional backbone
        for idx, stage in enumerate(self.base_model.module.stages):
            feature_map = stage(feature_map)
            if idx == self.num_stages - 2:
                break

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
        # Forward pass through the last stage
        feature_map_grad_l = self.base_model.module.norm_pre(self.base_model.module.stages[-1](feature_map_grad))
        # Forward pass through the head
        out = self.base_model.module.forward_head(feature_map_grad_l)
        return out

    def forward_feat(self, x: Tensor, train: bool = False) -> Tensor:
        # Forward pass through the convolutional backbone
        feature_map = self.base_model.stem(x)
        # Forward pass through the convolutional backbone
        for idx, stage in enumerate(self.base_model.stages):
            feature_map = stage(feature_map)
            if idx == self.num_stages - 2:
                break

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
        # Forward pass through the last stage
        feature_map_grad_l = self.base_model.norm_pre(self.base_model.stages[-1](feature_map_grad))
        # Forward pass through the head
        out = self.base_model.forward_head(feature_map_grad_l)
        return out
