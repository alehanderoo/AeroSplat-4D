# How feed-forward 3D Gaussian Splatting methods handle background color and prevent metric gaming

Your DepthSplat architecture is learning to output black because the target views have black backgrounds—a common pitfall when training object-centric reconstruction models. Analysis of **12 major feed-forward 3DGS papers** reveals that the standard solution is **not** using alpha-weighted RGB metrics, but rather a combination of **white backgrounds** and **separate alpha/mask supervision**. Here's how leading methods prevent exactly the problem you're facing.

## The universal approach: white backgrounds plus explicit mask losses

Every major Objaverse-trained method uses **white backgrounds** during training, not black. This is the critical first finding: LGM, GRM, TriplaneGaussian, Splatter Image, AGG, and others all render training data against white backgrounds. The rationale is straightforward—white backgrounds create maximum contrast with most object textures, making it harder for models to "cheat" by learning a uniform background color.

The second key technique is **explicit alpha/mask supervision as a separate loss term**. Rather than using alpha-weighted RGB metrics (which would let incorrect alpha predictions contaminate color learning), these methods supervise RGB and alpha channels with independent loss functions:

| Method | RGB Loss | Alpha/Mask Loss | Background Color |
|--------|----------|-----------------|------------------|
| **LGM** | MSE + LPIPS on full images | Separate MSE on alpha | White (fixed) |
| **GRM** | L2 + perceptual on full images | Separate L2 on mask | White |
| **TriplaneGaussian** | MSE on full images | Explicit L_MASK = \|\|M - M̂\|\|² | White (Blender) |
| **AGG** | L_rgba (joint) | Implicit via RGBA | White (Blender) |
| **Splatter Image** | MSE + LPIPS | Regularization losses | Configurable |
| **Gamba** | MSE + LPIPS | **Radial mask constraints** | White |

## Metrics are computed on full images, not foreground-only

A crucial finding: **none of these papers use alpha-weighted metrics for training**. All compute MSE, PSNR, SSIM, and LPIPS on the complete rendered image including background. The exact LGM loss formulation is:

```
ℒ_rgb = ℒ_MSE(I_rgb, I^GT_rgb) + λ·ℒ_LPIPS(I_rgb, I^GT_rgb)
ℒ_α = ℒ_MSE(I_α, I^GT_α)
```

The RGB and alpha losses are computed separately and summed. This prevents incorrect alpha predictions from reducing RGB loss—the model cannot improve its metrics by predicting "transparent where it should be opaque" because the alpha loss will penalize that independently.

For **evaluation**, Splatter Image documents an "Objects Only" protocol where background pixels in both predicted and ground-truth images are **set to black** before computing metrics. This ensures fair comparison without letting background accuracy inflate scores, but importantly both images use the same background color.

## Three specific mechanisms prevent background exploitation

**Mechanism 1: Consistent white backgrounds eliminate the incentive to game.** When both training renders and target views use white backgrounds, predicting uniform white would produce maximum error on object pixels (which have diverse colors), not minimum. GRM's paper states this explicitly for their Objaverse pipeline.

**Mechanism 2: Separate alpha loss forces correct silhouettes.** GRM's ablation study quantifies this effect:
- With alpha regularization: PSNR 29.48, SSIM 0.920, LPIPS 0.031
- Without alpha regularization: PSNR 29.38, SSIM 0.917, LPIPS 0.036

The α-reg term "removes floaters" and enforces accurate foreground boundaries. Without it, Gaussians can "fill in" background regions arbitrarily.

**Mechanism 3: Bounded Gaussian parameters prevent bloat.** GRM constrains Gaussian scales using a bounded sigmoid activation:
```
s = s_min × σ(s_o) + s_max × (1 - σ(s_o))
```
where s_min=0.005 and s_max=0.02. This prevents individual Gaussians from growing large enough to cover the entire background, forcing the model to predict transparency rather than large colored blobs.

## Gamba introduces radial mask constraints—most explicit solution

The Gamba paper (ECCV 2024) directly addresses foreground/background separation with **radial mask constraints** derived from multi-view masks. This technique explicitly prevents the model from "cheating" on background regions by enforcing geometric consistency between predicted alpha and known object silhouettes across views. This is the most explicit technique found for preventing exactly the problem you're experiencing.

LucidFusion takes a different approach: aggressive preprocessing with background removal and object recentering before training, ensuring the model never sees ambiguous background regions.

## Architectural features that implicitly help separation

**Pixel-aligned Gaussian prediction** (used by LGM, GRM, Splatter Image, pixelSplat, MVSplat) naturally helps background handling. When each pixel predicts a Gaussian constrained along its viewing ray, background pixels are forced to predict Gaussians at far depths with low opacity—they cannot easily produce colored "blobs" in the foreground.

Splatter Image takes this further: background pixels in the input image are **repurposed to encode occluded object regions**. The paper notes that "Splatter Images represent full 360° of objects by allocating background pixels to appropriate 3D locations." This architectural choice means background pixels contribute meaningfully to reconstruction rather than being wasted or exploited.

## What you should implement to fix DepthSplat

Based on this analysis, here are concrete recommendations for your architecture:

The most important change is switching from black to **white backgrounds** during both training data rendering and inference. Every successful Objaverse method uses white, not black. If your target views currently have black backgrounds, re-render them with white.

Second, add an **explicit mask loss** that supervises alpha/transparency separately from RGB:
```python
loss_rgb = F.mse_loss(pred_rgb, gt_rgb) + lambda_lpips * lpips_loss(pred_rgb, gt_rgb)
loss_mask = F.mse_loss(pred_alpha, gt_alpha)  # Separate term
loss_total = loss_rgb + lambda_mask * loss_mask
```

This prevents the model from trading off alpha accuracy for RGB accuracy—both must be correct independently.

Third, consider **bounded Gaussian scale activations** if you're not already using them. The GRM formulation (scales between 0.005 and 0.02) prevents Gaussians from covering large regions.

Finally, for evaluation, if you want "foreground-only" metrics, follow the Splatter Image protocol: set background pixels to the **same color** (they use black) in both prediction and ground truth before computing PSNR/SSIM/LPIPS. Never compute metrics where only one image has background substitution.

## Conclusion

The feed-forward 3DGS literature reveals a consistent approach: **white backgrounds** plus **separate alpha supervision** solves the background exploitation problem. No major method uses alpha-weighted RGB losses or foreground-masked training losses. The key insight is that decoupling RGB and alpha supervision prevents the model from gaming one to improve the other. Your model is learning black because black targets give it an easy path to low MSE—switching to white backgrounds removes that shortcut, and separate alpha loss ensures correct transparency regardless of background color.