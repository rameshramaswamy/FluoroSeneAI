import torch
from monai.networks.nets import UNet, SwinUNETR
from src.training.config_schema import TrainingConfig
from src.common.logger import configure_logger

log = configure_logger("model_factory")

class ModelFactory:
    @staticmethod
    def create_model(config: TrainingConfig):
        log.info("initializing_model", arch=config.model_architecture)
        
        if config.model_architecture == "unet":
            return UNet(
                spatial_dims=config.spatial_dims,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        
        elif config.model_architecture == "swin_unetr":
            # State-of-the-art Transformer for medical segmentation
            # Requires input size to be divisible by patch size
            return SwinUNETR(
                img_size=config.roi_size,
                in_channels=1,
                out_channels=2,
                feature_size=24, # Lightweight settings
                use_checkpoint=True,
                spatial_dims=config.spatial_dims
            )
        
        elif config.model_architecture == "temporal_unet":
            # 3D U-Net treating Time as Depth (Batch, C, Time, H, W)
            # Excellent for video consistency
            return UNet(
                spatial_dims=3, 
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        
        else:
            raise ValueError(f"Unknown architecture: {config.model_architecture}")