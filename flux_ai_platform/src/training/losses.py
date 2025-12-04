from monai.losses import DiceCELoss, DiceFocalLoss
from src.training.config_schema import TrainingConfig

def get_loss_function(config: TrainingConfig):
    if config.loss_function == "dice":
        return DiceCELoss(to_onehot_y=True, softmax=True)
    elif config.loss_function == "dice_focal":
        return DiceFocalLoss(
            to_onehot_y=True, 
            softmax=True, 
            gamma=config.focal_gamma, # Focuses on hard examples
            lambda_focal=1.0,
            lambda_dice=1.0
        )
    else:
        raise ValueError(f"Unknown loss: {config.loss_function}")