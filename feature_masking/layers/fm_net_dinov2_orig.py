import torch
from torch import Tensor
from .grad_mask import gradient_mask


# Baseline model, a modified ViT with feature masking after the last stage/block
class FeatureMaskingNetDinoV2Orig(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, seg_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.seg_model = seg_model
        self.backward_hook = None

    def forward_feat_ddp(self, x: Tensor, train: bool = False) -> Tensor:
        # Forward pass through the vit backbone
        feature_map = self.base_model.module.forward_features(x)
        cls_token = feature_map["x_norm_clstoken"] # [B, embed_dim]
        patch_tokens = feature_map["x_norm_patchtokens"]  # [B, num_patch_tokens, embed_dim]
        # Unflatten the feature map
        unflattened_fmap = self.base_model.module.unflatten(patch_tokens)  # [B, H, W, embed_dim]
        # Re-order the feature map to [B, embed_dim, H, W]
        unflattened_fmap = unflattened_fmap.permute(0, 3, 1, 2)
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
                unflattened_fmap.shape[-2], unflattened_fmap.shape[-1]), mode='nearest-exact')
            # Multiply the feature map with the segmentation mask
            unflattened_fmap.mul_(downsampled_mask)
        # Re-initialize the gradient and mask out the gradients of the feature map
        feature_map_grad, self.backward_hook = gradient_mask(unflattened_fmap, downsampled_mask, train)
        # Re-order the feature map back to [B, H, W, embed_dim]
        feature_map_grad = feature_map_grad.permute(0, 2, 3, 1)
        # Flatten the feature map back
        feature_map_grad = self.base_model.module.flatten_back(feature_map_grad)
        # Concatenate the cls token and the feature map
        feature_map_grad = torch.cat([cls_token.unsqueeze(1), feature_map_grad], dim=1)
        # Forward pass through the head
        out = self.base_model.module.head(feature_map_grad)
        return out

    def forward_feat(self, x: Tensor, train: bool = False) -> Tensor:
        # Forward pass through the vit backbone
        feature_map = self.base_model.forward_features(x)
        cls_token = feature_map["x_norm_clstoken"] # [B, embed_dim]
        patch_tokens = feature_map["x_norm_patchtokens"]  # [B, num_patch_tokens, embed_dim]
        # Unflatten the feature map
        unflattened_fmap = self.base_model.unflatten(patch_tokens)  # [B, H, W, embed_dim]
        # Re-order the feature map to [B, embed_dim, H, W]
        unflattened_fmap = unflattened_fmap.permute(0, 3, 1, 2)
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
                unflattened_fmap.shape[-2], unflattened_fmap.shape[-1]), mode='nearest-exact')
            # Multiply the feature map with the segmentation mask
            unflattened_fmap.mul_(downsampled_mask)
        # Re-initialize the gradient and mask out the gradients of the feature map
        feature_map_grad, self.backward_hook = gradient_mask(unflattened_fmap, downsampled_mask, train)
        # Re-order the feature map back to [B, H, W, embed_dim]
        feature_map_grad = feature_map_grad.permute(0, 2, 3, 1)
        # Flatten the feature map back
        feature_map_grad = self.base_model.flatten_back(feature_map_grad)
        # Concatenate the cls token and the feature map
        feature_map_grad = torch.cat([cls_token.unsqueeze(1), feature_map_grad], dim=1)
        # Forward pass through the head
        out = self.base_model.head(feature_map_grad)
        return out
