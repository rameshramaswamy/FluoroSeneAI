from monai.networks.nets import UNet
from src.training.config_schema import UNetConfig

def get_basic_unet(config: UNetConfig):
    return UNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
    )