import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedGate(nn.Module):
    """
    Mask-guided attention gate for emphasizing foreground boundaries.

    Based on research showing that soft gating with mask features
    improves depth boundary sharpness in feature fusion.
    """

    def __init__(self, features: int):
        super().__init__()
        # Project mask (1 channel) to feature space
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, features // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 4, features, 1),
        )
        # Gate signal from features + mask
        self.gate = nn.Sequential(
            nn.Conv2d(features * 2, features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask-guided gating to features.

        Args:
            features: Input features [B, C, H, W]
            mask: Mask tensor [B, 1, H, W] (will be resized if needed)

        Returns:
            Gated features [B, C, H, W]
        """
        # Resize mask to match feature resolution
        if mask.shape[-2:] != features.shape[-2:]:
            mask = F.interpolate(mask, size=features.shape[-2:], mode='bilinear', align_corners=True)

        # Project mask to feature space
        mask_features = self.mask_proj(mask)

        # Compute gate from concatenated features and mask features
        gate = self.gate(torch.cat([features, mask_features], dim=1))

        # Apply gating with residual connection
        return features * gate + features * 0.1


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block with optional mask-guided gating."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        use_mask_guided_gate=False,
    ):
        """Init.

        Args:
            features (int): number of features
            use_mask_guided_gate (bool): whether to use mask-guided attention gating
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.use_mask_guided_gate = use_mask_guided_gate

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

        # Optional mask-guided gate for skip connection
        if use_mask_guided_gate:
            self.mask_gate = MaskGuidedGate(features)

    def forward(self, *xs, size=None, mask=None):
        """Forward pass.

        Args:
            xs: Input features (1 or 2 tensors)
            size: Optional output size
            mask: Optional mask tensor for mask-guided gating [B, 1, H, W]

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            skip = xs[1]
            # Apply mask-guided gating to skip connection if enabled
            if self.use_mask_guided_gate and mask is not None:
                skip = self.mask_gate(skip, mask)
            res = self.resConfUnit1(skip)
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None, use_mask_guided_gate=False):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        use_mask_guided_gate=use_mask_guided_gate,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        concat_cnn_features=True,
        concat_mv_features=True,
        cnn_feature_channels=[64, 96, 128],
        concat_features=True,
        downsample_factor=8,
        return_feature=False,
        num_scales=1,
        use_mask_guided_gate=False,
        use_mask_boundary_upsample=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.concat_cnn_features = concat_cnn_features
        self.concat_mv_features = concat_mv_features
        self.concat_features = concat_features
        self.downsample_factor = downsample_factor
        self.return_feature = return_feature
        self.num_scales = num_scales
        self.use_mask_guided_gate = use_mask_guided_gate
        self.use_mask_boundary_upsample = use_mask_boundary_upsample

        if self.concat_features:
            if self.downsample_factor == 4 and num_scales == 2:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0] + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[1]
                            + out_channels[1]
                            + 64
                            + depth_channel,
                            out_channels[1],
                            1,
                        ),  # 1/4 concat(cnn, mono, mv, depth)
                        nn.Conv2d(
                            cnn_feature_channels[2] + out_channels[2] + 128,
                            out_channels[2],
                            1,
                        ),  # 1/8 concat(cnn, mono, mv)
                    ]
                )
            elif self.downsample_factor == 2 and num_scales == 2:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0]
                            + cnn_feature_channels[1]
                            + out_channels[0]
                            + 64
                            + depth_channel,
                            out_channels[0],
                            1,
                        ),  # 1/2
                        nn.Conv2d(
                            cnn_feature_channels[2] + out_channels[1] + 128,
                            out_channels[1],
                            1,
                        ),  # 1/4 concat(cnn, mono, mv, depth)
                        nn.Conv2d(out_channels[2], out_channels[2], 1),  # 1/8 mono
                    ]
                )
            elif self.downsample_factor == 4 and num_scales == 1:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0]
                            + cnn_feature_channels[1]
                            + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2]
                            + out_channels[1]
                            + 128
                            + depth_channel,
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(out_channels[2], out_channels[2], 1),  # 1/8 mono
                    ]
                )
            else:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0] + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[1] + out_channels[1],
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2]
                            + out_channels[2]
                            + 128
                            + depth_channel,
                            out_channels[2],
                            1,
                        ),  # 1/8 concat(cnn, mono, mv, depth)
                    ]
                )
        else:
            if self.concat_cnn_features:
                self.cnn_projects = nn.ModuleList(
                    [
                        nn.Conv2d(cnn_feature_channels[i], out_channels[i], 1)
                        for i in range(len(cnn_feature_channels))
                    ]
                )

            if self.concat_mv_features:
                self.mv_projects = nn.Conv2d(128, out_channels[2], 1)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn, use_mask_guided_gate=use_mask_guided_gate)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn, use_mask_guided_gate=use_mask_guided_gate)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, use_mask_guided_gate=use_mask_guided_gate)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, use_mask_guided_gate=False)  # No skip at this level

        # not used
        del self.scratch.refinenet4.resConfUnit1

        head_features_1 = features
        head_features_2 = 16

        if not self.return_feature:
            if use_mask_boundary_upsample:
                # Mask-boundary guided output conv with edge feature concatenation
                # Input: features (head_features_1) + mask (1) + mask_edges (1)
                self.scratch.output_conv_input = nn.Conv2d(
                    head_features_1 + 2,  # +2 for mask and mask edges
                    head_features_1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                )
                self.scratch.output_conv = nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(
                        head_features_1,
                        head_features_1 // 2,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        head_features_1 // 2,
                        head_features_2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="replicate",
                    ),
                    nn.GELU(),
                    nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                )
            else:
                self.scratch.output_conv = nn.Sequential(
                    nn.Conv2d(
                        head_features_1,
                        head_features_1 // 2,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        head_features_1 // 2,
                        head_features_2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="replicate",
                    ),
                    nn.GELU(),
                    nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                )

            # init delta depth as zero
            nn.init.zeros_(self.scratch.output_conv[-1].weight)
            nn.init.zeros_(self.scratch.output_conv[-1].bias)

    def forward(
        self,
        out_features,
        downsample_factor=8,
        cnn_features=None,
        mv_features=None,
        depth=None,
        mask=None,
    ):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        # 1/2, 1/4, 1/8, 1/16
        layer_1, layer_2, layer_3, layer_4 = out

        if self.concat_features:
            if not self.return_feature:
                assert depth is not None

            if self.downsample_factor == 4 and self.num_scales == 1:
                concat1 = torch.cat((cnn_features[0], cnn_features[1], layer_1), dim=1)
            elif self.downsample_factor == 2 and self.num_scales == 2:
                if self.return_feature:
                    concat1 = torch.cat(
                        (cnn_features[0], cnn_features[1], mv_features[0], layer_1),
                        dim=1,
                    )
                else:
                    concat1 = torch.cat(
                        (
                            cnn_features[0],
                            cnn_features[1],
                            mv_features[0],
                            depth,
                            layer_1,
                        ),
                        dim=1,
                    )
            else:
                concat1 = torch.cat((cnn_features[0], layer_1), dim=1)
            layer_1 = self.concat_projects[0](concat1)  # 1/2

            if self.downsample_factor == 4 and self.num_scales == 2:
                assert isinstance(mv_features, list)
                if self.return_feature:
                    concat2 = torch.cat(
                        (cnn_features[1], layer_2, mv_features[0]), dim=1
                    )
                else:
                    concat2 = torch.cat(
                        (cnn_features[1], layer_2, mv_features[0], depth), dim=1
                    )
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                concat3 = torch.cat((cnn_features[2], layer_3, mv_features[1]), dim=1)
                layer_3 = self.concat_projects[2](concat3)  # 1/8
            elif self.downsample_factor == 2 and self.num_scales == 2:
                assert isinstance(mv_features, list)
                concat2 = torch.cat((cnn_features[2], layer_2, mv_features[1]), dim=1)
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                concat3 = layer_3
                layer_3 = self.concat_projects[2](concat3)  # 1/8
            elif self.downsample_factor == 4 and self.num_scales == 1:
                if self.return_feature:
                    concat2 = torch.cat((cnn_features[2], layer_2, mv_features), dim=1)
                else:
                    concat2 = torch.cat(
                        (cnn_features[2], layer_2, mv_features, depth), dim=1
                    )
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                concat3 = layer_3
                layer_3 = self.concat_projects[2](concat3)  # 1/8
            else:
                concat2 = torch.cat((cnn_features[1], layer_2), dim=1)
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                if self.return_feature:
                    concat3 = torch.cat((cnn_features[2], layer_3, mv_features), dim=1)
                else:
                    concat3 = torch.cat(
                        (cnn_features[2], layer_3, mv_features, depth), dim=1
                    )
                layer_3 = self.concat_projects[2](concat3)  # 1/8
        else:
            if self.concat_cnn_features:
                assert cnn_features is not None
                assert len(cnn_features) == 3  # 1/2, 1/4, 1/8
                cnn_features = [
                    self.cnn_projects[i](f) for i, f in enumerate(cnn_features)
                ]

                layer_1 = layer_1 + cnn_features[0]  # 1/2
                layer_2 = layer_2 + cnn_features[1]  # 1/4
                layer_3 = layer_3 + cnn_features[2]  # 1/8

            if self.concat_mv_features:
                # 1/8
                mv_features = self.mv_projects(mv_features)

                layer_3 = layer_3 + mv_features  # 1/8

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])  # 1/8
        path_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:], mask=mask
        )  # 1/4
        path_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:], mask=mask
        )  # 1/2
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, mask=mask)  # 1

        if self.return_feature:
            return path_1

        if self.use_mask_boundary_upsample and mask is not None:
            # Resize mask to match path_1 resolution
            mask_resized = F.interpolate(mask, size=path_1.shape[-2:], mode='bilinear', align_corners=True)

            # Compute mask edges using Sobel filter
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=path_1.dtype, device=path_1.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=path_1.dtype, device=path_1.device).view(1, 1, 3, 3)
            edge_x = F.conv2d(mask_resized, sobel_x, padding=1)
            edge_y = F.conv2d(mask_resized, sobel_y, padding=1)
            mask_edges = (edge_x.pow(2) + edge_y.pow(2)).sqrt()

            # Concatenate features with mask and mask edges
            path_1_with_mask = torch.cat([path_1, mask_resized, mask_edges], dim=1)

            # Apply mask-boundary guided output conv
            path_1_processed = self.scratch.output_conv_input(path_1_with_mask)
            out = self.scratch.output_conv(path_1_processed)
        else:
            out = self.scratch.output_conv(path_1)

        return out


if __name__ == "__main__":
    device = torch.device("cuda")
    c = 384
    model = DPTHead(
        in_channels=c,
        concat_cnn_features=True,
        concat_mv_features=True,
    ).to(device)
    print(model)

    h, w = 16, 32

    x = torch.randn(2, c, h, w).to(device)

    out_features = [x] * 4

    cnn_features = [
        torch.randn(2, 64, h * 4, w * 4).to(device),
        torch.randn(2, 96, h * 2, w * 2).to(device),
        torch.randn(2, 128, h, w).to(device),
    ]

    mv_features = torch.randn(2, 128, h, w).to(device)

    out = model(out_features, h, w, cnn_features=cnn_features, mv_features=mv_features)

    print(out.shape)
