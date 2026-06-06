import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisualEncoder(nn.Module):
    """
    RGB student encoder that maps the final image into the teacher projection space.
    """

    def __init__(
        self,
        d_out: int = 256,
        backbone_type: str = "resnet50",
        pretrained: bool = True,
        num_stages: int = 5,
        input_mode: str = "rgb",
        residual_scale: float = 1.0,
        residual_kernel: int = 5,
        use_stage_sequence_head: bool = False,
        local_crop_mode: str = "none",
        local_crop_size: int = 160,
        local_patch_offset: int = 24,
    ):
        super().__init__()
        self.d_out = int(d_out)
        self.num_stages = int(num_stages)
        self.feature_dim = 0
        self.input_mode = str(input_mode)
        self.residual_scale = float(residual_scale)
        self.residual_kernel = int(residual_kernel)
        self.use_stage_sequence_head = bool(use_stage_sequence_head)
        self.local_crop_mode = str(local_crop_mode)
        self.local_crop_size = int(local_crop_size)
        self.local_patch_offset = int(local_patch_offset)
        if self.residual_kernel % 2 == 0:
            raise ValueError("residual_kernel must be odd.")

        if self.input_mode in {"rgb", "residual_only"}:
            input_channels = 3
        elif self.input_mode == "rgb_residual":
            input_channels = 6
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        if backbone_type == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            feature_dim = resnet.fc.in_features
        elif backbone_type == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            feature_dim = resnet.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        self.feature_dim = int(feature_dim)

        if input_channels != 3:
            conv1 = resnet.conv1
            new_conv1 = nn.Conv2d(
                input_channels,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv1.weight[:, :3].copy_(conv1.weight)
                extra_channels = input_channels - 3
                extra_weight = conv1.weight.mean(dim=1, keepdim=True).repeat(1, extra_channels, 1, 1)
                new_conv1.weight[:, 3:].copy_(extra_weight)
            resnet.conv1 = new_conv1

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, self.d_out),
        )
        if self.local_crop_mode == "none":
            self.local_attention = None
            self.local_fusion = None
        elif self.local_crop_mode == "center_patch5":
            self.local_attention = nn.Linear(feature_dim, 1)
            self.local_fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported local_crop_mode: {self.local_crop_mode}")
        if self.use_stage_sequence_head:
            self.sequence_projector = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.BatchNorm1d(feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 2, self.num_stages * self.d_out),
            )
        else:
            self.sequence_projector = None

    def _forward_backbone(self, image_tensor: torch.Tensor):
        spatial_map = image_tensor
        for layer in self.backbone[:-1]:
            spatial_map = layer(spatial_map)
        pooled_features = self.backbone[-1](spatial_map)
        pooled_features = torch.flatten(pooled_features, 1)
        return spatial_map, pooled_features

    def _encode_backbone_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        _, features = self._forward_backbone(image_tensor)
        return features

    def _project_features(self, feature_tensor: torch.Tensor, return_logits: bool = False):
        feature_shape = feature_tensor.shape[:-1]
        flat_features = feature_tensor.reshape(-1, feature_tensor.shape[-1])
        projected_logits = self.projector(flat_features)
        projected_logits = projected_logits.reshape(*feature_shape, self.d_out)
        projected_repr = F.normalize(projected_logits, dim=-1)
        if return_logits:
            return projected_repr, projected_logits
        return projected_repr

    def _extract_local_crops(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if self.local_crop_mode == "none":
            return None
        _, _, height, width = image_tensor.shape
        crop_size = min(self.local_crop_size, height, width)
        max_y = max(height - crop_size, 0)
        max_x = max(width - crop_size, 0)
        center_y = max_y // 2
        center_x = max_x // 2
        delta_y = min(self.local_patch_offset, center_y, max_y - center_y)
        delta_x = min(self.local_patch_offset, center_x, max_x - center_x)
        coords = [
            (center_y, center_x),
            (max(center_y - delta_y, 0), max(center_x - delta_x, 0)),
            (max(center_y - delta_y, 0), min(center_x + delta_x, max_x)),
            (min(center_y + delta_y, max_y), max(center_x - delta_x, 0)),
            (min(center_y + delta_y, max_y), min(center_x + delta_x, max_x)),
        ]
        crops = []
        for top, left in coords:
            crops.append(image_tensor[:, :, top : top + crop_size, left : left + crop_size])
        return torch.stack(crops, dim=1)

    def forward(
        self,
        rgb_image: torch.Tensor,
        return_sequence: bool = False,
        return_patch_tokens: bool = False,
        return_logits: bool = False,
        return_features: bool = False,
    ):
        is_multiview = rgb_image.ndim == 5
        if is_multiview:
            batch_size, num_views, channels, height, width = rgb_image.shape
            rgb_image = rgb_image.reshape(batch_size * num_views, channels, height, width)

        if self.input_mode in {"rgb_residual", "residual_only"}:
            padding = self.residual_kernel // 2
            blurred = F.avg_pool2d(rgb_image, kernel_size=self.residual_kernel, stride=1, padding=padding)
            residual = rgb_image - blurred
            if self.input_mode == "rgb_residual":
                rgb_image = torch.cat([rgb_image, self.residual_scale * residual], dim=1)
            else:
                rgb_image = self.residual_scale * residual

        spatial_map, features = self._forward_backbone(rgb_image)
        if self.local_crop_mode != "none":
            local_crops = self._extract_local_crops(rgb_image)
            local_crops = local_crops.reshape(-1, local_crops.shape[2], local_crops.shape[3], local_crops.shape[4])
            local_features = self._encode_backbone_features(local_crops)
            local_features = local_features.reshape(features.shape[0], -1, local_features.shape[-1])
            local_attn = torch.softmax(self.local_attention(local_features).squeeze(-1), dim=1)
            local_agg = (local_features * local_attn.unsqueeze(-1)).sum(dim=1)
            features = self.local_fusion(torch.cat([features, local_agg], dim=-1))

        if is_multiview:
            features = features.reshape(batch_size, num_views, -1).mean(dim=1)
            spatial_map = spatial_map.reshape(batch_size, num_views, *spatial_map.shape[1:]).mean(dim=1)
        global_repr, global_logits = self._project_features(features, return_logits=True)
        if return_sequence or return_patch_tokens or return_logits or return_features:
            outputs = {"global_repr": global_repr}
            if return_logits:
                outputs["global_logits"] = global_logits
            if return_features:
                outputs["feature_vec"] = features
            if self.sequence_projector is not None:
                stage_logits = self.sequence_projector(features).reshape(features.shape[0], self.num_stages, self.d_out)
                stage_repr = F.normalize(stage_logits, dim=-1)
                outputs["stage_repr"] = F.normalize(stage_repr, dim=-1)
                if return_logits:
                    outputs["stage_logits"] = stage_logits
            if return_patch_tokens:
                patch_features = spatial_map.flatten(2).transpose(1, 2)
                outputs["patch_repr"] = self._project_features(patch_features)
            return outputs
        return global_repr
