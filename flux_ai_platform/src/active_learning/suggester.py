import torch
import numpy as np
from src.training.trainer import FluxTrainer
from src.training.config_schema import UNetConfig
from src.common.logger import configure_logger

log = configure_logger("active_learning")

class AnnotationSuggester:
    def __init__(self, model_path: str = None):
        self.config = UNetConfig()
        self.trainer = FluxTrainer(config=self.config)
        
        if model_path:
            try:
                # Load strictly onto the trainer's device
                state_dict = torch.load(model_path, map_location=self.trainer.device)
                self.trainer.model.load_state_dict(state_dict)
            except Exception:
                log.warning("model_load_failed_using_random", path=model_path)
        
        self.trainer.model.eval()

    def predict_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """
        Input: Tensor (B, 1, H, W) on Device
        Output: Numpy (B, H, W) - Segmentation Masks
        """
        with torch.no_grad():
            # Ensure input is on correct device
            if batch_tensor.device != self.trainer.device:
                batch_tensor = batch_tensor.to(self.trainer.device)

            logits = self.trainer.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_masks = torch.argmax(probs, dim=1)
            
            # Return as uint8 numpy array
            return pred_masks.cpu().numpy().astype(np.uint8)
        

    def predict_batch_with_uncertainty(self, batch_tensor: torch.Tensor):
        """
        Returns: 
        - masks: (B, H, W)
        - uncertainty_scores: (B,) float value 0-1
        """
        with torch.no_grad():
            if batch_tensor.device != self.trainer.device:
                batch_tensor = batch_tensor.to(self.trainer.device)

            logits = self.trainer.model(batch_tensor)
            probs = torch.softmax(logits, dim=1) # (B, C, H, W)
            pred_masks = torch.argmax(probs, dim=1) # (B, H, W)
            
            # Calculate Entropy Map: -sum(p * log(p))
            # We average the entropy over the image pixels to get a scalar score
            # High Score = High Uncertainty
            entropy_map = -torch.sum(probs * torch.log(probs + 1e-6), dim=1) # (B, H, W)
            uncertainty_scores = torch.mean(entropy_map, dim=(1, 2)) # (B,)
            
            return (
                pred_masks.cpu().numpy().astype(np.uint8), 
                uncertainty_scores.cpu().numpy()
            )