from pydantic import BaseModel, Field
from typing import Tuple, Optional

class UNetConfig(BaseModel):
    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 2
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    
    class Config:
        frozen = True # Immutable config



class TrainingConfig(BaseModel):
    # Experiment Metadata
    experiment_name: str = "flux_fluoroscopy_segmentation"
    run_name: Optional[str] = None
    
    # Model Architecture
    model_architecture: str = "unet" # Options: "unet", "swin_unetr", "temporal_unet"
    spatial_dims: int = 2 # 2 for single frame, 3 for video volume
    
    # Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_epochs: int = 10
    roi_size: Tuple[int, int] = (96, 96) # Patch size for Swin
    
    # Loss Configuration
    loss_function: str = "dice_focal" # Options: "dice", "dice_focal"
    focal_gamma: float = 2.0
    
    class Config:
        frozen = False # Allow Optuna to modify values