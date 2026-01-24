# Mask integration strategies for multi-view stereo depth estimation

Object segmentation masks represent an **underutilized prior** in your current DepthSplat architecture. Research across MVS, neural implicit methods, and 3D Gaussian splatting reveals that masks should be integrated at **six distinct pipeline stages** beyond simple background zeroing—with soft weighting and learned integration consistently outperforming binary operations.

Your current approach (zeroing background cost volume, forcing far-plane background depth) implements only the most basic mask usage. The literature demonstrates that carefully designed mask integration at the feature extraction, attention, cost aggregation, regularization, and upsampling stages yields substantial improvements in depth boundary sharpness and reconstruction quality. The key insight: **soft mask integration with preserved gradients outperforms hard binary masking** at every stage.

## Cost volume construction requires uncertainty-weighted soft masking

Binary cost volume zeroing creates gradient discontinuities and discards valid texture information near object boundaries. The proven alternative from Vis-MVSNet and DCV-MVSNet uses **soft confidence weighting**:

$$C_{\text{weighted}}(p,d) = \frac{\sum_i w_i(p) \cdot C_i(p,d)}{\sum_i w_i(p) + \epsilon}$$

where weights $w_i$ derive from learned uncertainty or mask confidence. PatchmatchNet demonstrates that pixel-wise view weights, learned jointly with matching, outperform hand-crafted weighting. For object-centric scenarios, the mask provides a strong prior for this weighting:

```python
def soft_weighted_cost_volume(features_ref, features_src, mask, depths):
    # Soft weight: high confidence in foreground, uncertainty at boundaries
    mask_confidence = mask_to_confidence(mask)  # sigmoid, not binary
    
    for i, feat_src in enumerate(features_src):
        warped = homography_warp(feat_src, depths, cameras[i])
        cost = compute_variance_cost(features_ref, warped)
        
        # Soft weighting preserves gradients
        cost_volume += mask_confidence * cost
        weight_sum += mask_confidence
    
    return cost_volume / (weight_sum + 1e-8)
```

The **temperature-scaled approach** offers finer control: $w_i = \exp(s_i/\tau) / \sum_j \exp(s_j/\tau)$, where τ→0 approaches binary masking. This provides a learnable interpolation between soft and hard masking.

For edge preservation specifically, EPNet's hierarchical edge-preserving residual module and EG-MVSNet's dynamic Sobel kernels demonstrate that edge-aware cost regularization prevents the 3D CNN from over-smoothing object boundaries. Consider adding an edge feature volume that's adaptively fused with the standard cost volume:

$$C_{\text{fused}} = \sigma(\alpha) \cdot C_{\text{standard}} + (1-\sigma(\alpha)) \cdot C_{\text{edge}}$$

## Transformer attention benefits from asymmetric foreground-background masking

For your multi-view transformer with cross-view attention, the mathematically correct approach uses **additive masking before softmax**:

$$\text{Attention}(Q,K,V,M) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

where $M_{ij} = 0$ for valid attention positions and $M_{ij} = -\infty$ for blocked positions. This preserves proper probability normalization and maintains gradient flow through unmasked positions. Multiplicative masking after softmax breaks normalization and creates unstable gradients.

The critical design question—should foreground attend to background?—has a research-backed answer: **asymmetric masking with separate attention heads**. CamoFormer's Masked Separable Attention allocates heads to three groups: foreground-only attention, background-only attention, and global context attention. For your architecture:

```python
class MaskedCrossViewAttention(nn.Module):
    def forward(self, query, key, value, object_mask, epipolar_mask=None):
        # Combine geometric (epipolar) and semantic (object) constraints
        combined_mask = torch.zeros_like(query @ key.T)
        invalid = ~object_mask if self.fg_only else object_mask
        combined_mask[invalid] = float('-inf')
        
        if epipolar_mask is not None:
            combined_mask[~epipolar_mask] = float('-inf')
        
        scores = query @ key.T / math.sqrt(self.d_k) + combined_mask
        attn = F.softmax(scores, dim=-1)
        return attn @ value + query  # residual connection critical
```

MVSTER's epipolar transformer demonstrates that constraining attention to geometrically valid correspondences (along epipolar lines) dramatically reduces computation while improving accuracy. For object-centric reconstruction, combining epipolar constraints with object mask constraints yields the tightest correspondence search space.

**Gradient stability checklist**: (1) scale QK before masking to prevent softmax saturation, (2) use residual connections to ensure gradient paths even when attention is sparse, (3) maintain **10-20% of attention heads as global** (unmasked) to preserve scene context.

## Object-centric neural methods reveal occlusion-aware mask supervision

ObjectSDF++ provides the most sophisticated mask supervision framework. The key insight: simply matching rendered mask to ground truth ignores occlusion. The **occlusion-aware opacity rendering** formulation correctly handles objects behind other objects:

$$\hat{O}_{O_i}(r) = \int_{v_n}^{v_f} T_\Omega(v) \cdot \sigma_{O_i}(r(v)) dv$$

where $T_\Omega$ is scene transmittance (cumulative product of all object transmittances along the ray), and $\sigma_{O_i}$ is the density of object $i$. This ensures the frontal object absorbs all light along the ray.

For depth estimation specifically, ObjectSDF++'s **object distinction regularization** prevents objects from growing into each other in unobserved regions:

$$L_{\text{reg}} = \mathbb{E}_p\left[\sum_{d_{O_i}(p) \neq d_\Omega(p)} \text{ReLU}(d_{O_i}(p) - d_\Omega(p))\right]$$

This loss penalizes when a non-surface-closest object's SDF extends beyond the scene SDF.

The recommended loss combination from this literature:

| Loss Component | Weight | Purpose |
|---------------|--------|---------|
| Depth L1/L2 | 1.0 | Primary supervision |
| Mask BCE/L2 | 1.0 | Foreground separation |
| Gradient matching | 0.1-0.5 | Edge sharpness |
| Eikonal regularization | 0.1 | Smooth depth gradients |
| Object distinction | 0.5 | Prevent object bleeding |

## Feature-level mask integration should use soft gating with residuals

The research consensus advises **against** hard multiplication of masks with features before cost volume construction—this zeros out valid texture information at boundaries. Instead, use soft multiplicative gating with a residual pathway:

```python
def mask_guided_features(features, mask):
    # Learn whether to apply mask vs preserve original
    gate = sigmoid(mask_projection(mask))
    return gate * features * mask + (1 - gate) * features
```

**Pixel-Adaptive Convolution (PAC)** provides a principled framework for mask-guided feature extraction:

$$v'_i = \sum_{j \in \Omega(i)} K(f_i, f_j) \cdot W[p_i - p_j] \cdot v_j + b$$

where $K(f_i, f_j) = \exp(-\frac{1}{2}\|f_i - f_j\|^2)$ is an adapting kernel that can incorporate mask boundary features. This allows the convolution to adapt its behavior based on whether pixels are in the same mask region.

For your **CNN + DINOv2 dual-branch architecture**, the recommended integration points:

1. **CNN branch**: Inject mask features via lateral connections in FPN-style skip connections
2. **DINOv2 branch**: Concatenate mask tokens as additional input tokens, or use cross-attention between image and mask tokens
3. **Feature fusion**: Apply mask-guided attention gates when combining the two branches

```python
# Attention gate with mask guidance (in skip connections)
gate_signal = conv(encoder_features)
mask_signal = conv(mask_features)
attention = sigmoid(conv(relu(gate_signal + mask_signal)))
output = decoder_features * attention
```

## DPT upsampling requires mask-guided refinement at multiple scales

Your DPT upsampler with residual refinement has three natural mask integration points:

**Point 1 - Reassemble stage**: After converting ViT tokens to 2D feature maps, multiply by mask-derived attention:
```python
reassembled = reassemble(tokens)
mask_attention = sigmoid(conv(upsample(mask, reassembled.shape)))
reassembled = reassembled * mask_attention + reassembled * 0.1  # residual
```

**Point 2 - Fusion blocks**: The RefineNet-style fusion uses residual convolutions. Insert mask-guided skip weighting:
```python
class MaskGuidedFusion(nn.Module):
    def forward(self, features, skip, mask):
        mask_gate = sigmoid(self.mask_proj(mask))
        gated_skip = skip * mask_gate  # Emphasize foreground boundaries
        return self.residual_conv(features + gated_skip)
```

**Point 3 - Final upsampling**: Replace bilinear upsampling with **PAC-transposed convolution** using mask boundaries as guidance:
```python
class MaskGuidedUpsampler(nn.Module):
    def forward(self, depth_lr, mask):
        mask_edges = gradient_magnitude(mask)
        guide = torch.cat([mask, mask_edges], dim=1)
        return self.pac_transpose(depth_lr, guide)  # 2x upsample
```

For edge preservation, Depth Anything V2's **gradient matching loss** at multiple scales is essential:

$$L_{\text{gm}} = \sum_s \|\nabla_x(d_{\text{pred}}^s) - \nabla_x(d_{\text{gt}}^s)\|_1 + \|\nabla_y(d_{\text{pred}}^s) - \nabla_y(d_{\text{gt}}^s)\|_1$$

Weight this loss more heavily at mask boundaries: $L_{\text{edge}} = L_{\text{gm}} \cdot \text{dilate}(\text{edge}(M))$.

## 3D Gaussian initialization requires pixel-aligned depth with alpha supervision

For your downstream Gaussian reconstruction, the research from GRM and LGM is definitive: **pixel-aligned Gaussian placement along viewing rays significantly outperforms direct XYZ prediction**:

$$\mu = c_o + \tau \cdot r$$

where $c_o$ is the camera center, $\tau$ is predicted depth, and $r = K^{-1}[u,v,1]^\top$ is the ray direction. This constrains Gaussians to geometrically valid positions and provides structural regularization that prevents local minima.

**Alpha mask supervision** during training is essential for object-centric reconstruction:

$$L_{\text{mask}} = \|M_{\text{rendered}} - M_{\text{GT}}\|_2^2$$

This loss removes "floater" Gaussians outside the object bounds that would otherwise accumulate during optimization. GRM and LGM both report this as critical for clean reconstructions.

The **depth accuracy requirements** for good Gaussian initialization are surprisingly lenient—**relative depth ordering matters more than absolute accuracy**. Scale-shift alignment to sparse SfM points is standard practice:

$$D_{\text{aligned}} = a \cdot D_{\text{mono}} + b, \quad \text{where } (a,b) = \argmin \sum_i \|a \cdot D_i + b - D_{\text{SfM}}\|^2$$

However, **depth boundary sharpness directly impacts Gaussian placement quality**. Soft depth edges create elongated Gaussians straddling foreground and background. SAGD's Gaussian Decomposition technique specifically addresses this by identifying and splitting boundary Gaussians.

## Recommended architecture modifications

Based on the synthesized research, here are the specific changes to your DepthSplat architecture, ordered by expected impact:

**High impact, moderate complexity:**
1. Replace binary cost volume zeroing with soft confidence weighting derived from mask
2. Add alpha mask supervision loss ($L_{\text{mask}} = \|M_{\text{rendered}} - M_{\text{GT}}\|_2^2$)
3. Add gradient matching loss weighted at mask boundaries
4. Use additive masking ($-\infty$) in transformer cross-attention, not multiplicative

**Medium impact, lower complexity:**
5. Add mask-guided attention gates in DPT fusion blocks
6. Implement PAC-style upsampling with mask boundary features as guidance
7. Split attention heads: 70% foreground-only, 20% background-only, 10% global

**Lower impact, experimental:**
8. Concatenate mask tokens to DINOv2 input
9. Object distinction regularization loss (prevents depth bleeding)
10. Uncertainty-guided dynamic cost volume blending (pixel vs patch costs)

The mathematical formulations are gradient-friendly when using soft operations (sigmoid gating, additive attention masking, temperature-scaled softmax). Avoid hard binary masks in the forward pass—use them only for loss computation or convert to soft confidence scores for feature operations. This ensures stable training while still leveraging the strong object boundary prior that masks provide.

## Conclusion

Your current mask usage captures only a fraction of available signal. The literature demonstrates that masks should inform **every stage** of the depth estimation pipeline: feature extraction (soft gating), attention (additive masking with separate FG/BG heads), cost volume (confidence weighting), regularization (edge-weighted losses), and upsampling (PAC-guided refinement). The consistent theme across methods—from MVSNet variants to ObjectSDF++ to GRM—is that **learned, soft mask integration outperforms hand-crafted binary operations**, both in reconstruction quality and training stability. For Objaverse-style object-centric datasets, these modifications should yield sharper depth boundaries, cleaner background separation, and ultimately higher-quality 3D Gaussian reconstructions.