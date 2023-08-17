import torch
from torch import Tensor
from .grad_mask import gradient_mask


# Baseline model, a modified ViT with feature masking after the second last stage/block
class FeatureMaskingNetDinoV2SecondLastBlock(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, seg_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.seg_model = seg_model
        self.backward_hook = None
        self.num_stages = len(self.base_model.blocks)

    def forward_feat_ddp(self, inp_tensor: Tensor, train: bool = False) -> Tensor:
        # Create the positional and patch embeddings
        x = self.base_model.module.patch_embed(inp_tensor)
        x = self.base_model.module._pos_embed(x)
        x = self.base_model.module.patch_drop(x)
        x = self.base_model.module.norm_pre(x)
        # Forward pass through the attention blocks
        for idx, blk in enumerate(self.base_model.module.blocks):
            x = blk(x)
            if idx == self.num_stages - 2:
                break
        # Apply additional layer normalization
        x = self.base_model.module.norm_sl(x)  # [B, num_patch_tokens + 1, embed_dim]
        # Split the CLS token and the patch tokens
        cls_token = x[:, 0, :]  # [B, embed_dim]
        patch_tokens = x[:, 1:, :]  # [B, num_patch_tokens, embed_dim]
        # Unflatten the feature map
        unflattened_fmap = self.base_model.module.unflatten(patch_tokens)  # [B, H, W, embed_dim]
        # Re-order the feature map to [B, embed_dim, H, W]
        unflattened_fmap = unflattened_fmap.permute(0, 3, 1, 2)
        with torch.no_grad():
            batch_img_metas = [
                                  dict(
                                      ori_shape=inp_tensor.shape[2:],
                                      img_shape=inp_tensor.shape[2:],
                                      pad_shape=inp_tensor.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inp_tensor.shape[0]
            outputs_seg = torch.argmax(self.seg_model.module.inference(inp_tensor, batch_img_metas), dim=1,
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
        # Forward pass through the last stage/block
        feature_map_grad = self.base_model.module.blocks[-1](feature_map_grad)
        # Layer norm
        feature_map_grad = self.base_model.module.norm(feature_map_grad)
        out_norm = self.base_model.module.fc_norm(feature_map_grad)
        out_head_drop = self.base_model.module.head_drop(out_norm)
        out = self.base_model.module.head(out_head_drop)
        return out

    def forward_feat(self, inp_tensor: Tensor, train: bool = False) -> Tensor:
        # Create the positional and patch embeddings
        x = self.base_model.patch_embed(inp_tensor)
        x = self.base_model._pos_embed(x)
        x = self.base_model.patch_drop(x)
        x = self.base_model.norm_pre(x)
        # Forward pass through the attention blocks
        for idx, blk in enumerate(self.base_model.blocks):
            x = blk(x)
            if idx == self.num_stages - 2:
                break
        cls_token = x[:, 0, :]
        patch_tokens = x[:, 1:, :]  # [B, num_patch_tokens, embed_dim]
        # Unflatten the feature map
        unflattened_fmap = self.base_model.unflatten(patch_tokens)  # [B, H, W, embed_dim]
        # Re-order the feature map to [B, embed_dim, H, W]
        unflattened_fmap = unflattened_fmap.permute(0, 3, 1, 2)
        with torch.no_grad():
            batch_img_metas = [
                                  dict(
                                      ori_shape=inp_tensor.shape[2:],
                                      img_shape=inp_tensor.shape[2:],
                                      pad_shape=inp_tensor.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inp_tensor.shape[0]
            outputs_seg = torch.argmax(self.seg_model.inference(inp_tensor, batch_img_metas), dim=1,
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
        # Forward pass through the last stage/block
        feature_map_grad = self.base_model.blocks[-1](feature_map_grad)
        # Layer norm
        feature_map_grad = self.base_model.norm(feature_map_grad)
        # Forward pass through the head
        out_norm = self.base_model.fc_norm(feature_map_grad)
        out_head_drop = self.base_model.head_drop(out_norm)
        out = self.base_model.head(out_head_drop)
        return out
