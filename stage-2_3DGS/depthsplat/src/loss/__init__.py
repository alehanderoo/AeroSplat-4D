from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_mask import LossMask, LossMaskCfgWrapper
from .loss_gradient import LossGradient, LossGradientCfgWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossDepthCfgWrapper: LossDepth,
    LossMaskCfgWrapper: LossMask,
    LossGradientCfgWrapper: LossGradient,
}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper | LossDepthCfgWrapper | LossMaskCfgWrapper | LossGradientCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
