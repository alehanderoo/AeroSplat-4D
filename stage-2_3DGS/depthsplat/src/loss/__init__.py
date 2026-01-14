from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_silhouette import LossSilhouette, LossSilhouetteCfgWrapper
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_mask import LossMask, LossMaskCfgWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossSilhouetteCfgWrapper: LossSilhouette,
    LossDepthCfgWrapper: LossDepth,
    LossMaskCfgWrapper: LossMask,
}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper | LossSilhouetteCfgWrapper | LossDepthCfgWrapper | LossMaskCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
