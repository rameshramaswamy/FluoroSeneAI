import torch
import kornia
import numpy as np
from src.common.logger import configure_logger

log = configure_logger("preprocessing_service")

class ImageEnhancer:
    def __init__(self, target_size=(256, 256), use_gpu=True):
        """
        Initializes the Image Enhancer with GPU support.
        
        Args:
            target_size (tuple): Target (H, W) for the AI model.
            use_gpu (bool): Flag to enable CUDA if available.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.target_size = target_size
        
        log.info("enhancer_initialized", device=str(self.device), target_size=target_size)

    def preprocess_batch(self, images: np.ndarray) -> torch.Tensor:
        """
        Applies the full preprocessing pipeline on a batch of images using GPU acceleration.
        Pipeline: Normalize -> Resize -> Denoise -> CLAHE.

        Args:
            images (np.ndarray): Input batch of shape (Batch, H, W) or (Batch, C, H, W) or (H, W).
                                 Expects uint8 (0-255) or float (0-1).
        
        Returns:
            torch.Tensor: Preprocessed batch (Batch, 1, H, W) on GPU, normalized 0-1.
        """
        try:
            # 1. Convert to Tensor & Normalize to 0-1
            if isinstance(images, np.ndarray):
                img_t = torch.tensor(images)
            else:
                img_t = images

            # Handle Dimensions
            # Target shape for Kornia: (B, C, H, W)
            if img_t.ndim == 2:   # (H, W) -> (1, 1, H, W)
                img_t = img_t.unsqueeze(0).unsqueeze(0)
            elif img_t.ndim == 3: # (B, H, W) -> (B, 1, H, W)
                img_t = img_t.unsqueeze(1)
            # else: (B, C, H, W) assumed correct

            img_t = img_t.float().to(self.device)

            # Normalize 0-255 to 0-1 if necessary
            if img_t.max() > 1.0:
                img_t = img_t / 255.0

            # 2. Resize
            # Kornia resize expects (B, C, H, W)
            img_resized = kornia.geometry.resize(
                img_t, 
                self.target_size, 
                interpolation='bilinear', 
                antialias=True
            )

            # 3. Denoise (Gaussian Blur)
            # Using Gaussian Blur as a fast, differentiable proxy for Non-Local Means on GPU.
            # Kernel size (5,5) and sigma (1.5) are tuned for standard fluoroscopy noise.
            img_denoised = kornia.filters.gaussian_blur2d(
                img_resized, 
                kernel_size=(5, 5), 
                sigma=(1.5, 1.5)
            )

            # 4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Crucial for visualizing guidewires in low-contrast X-ray.
            img_enhanced = kornia.enhancement.equalize_clahe(
                img_denoised, 
                clip_limit=2.0, 
                grid_size=(8, 8)
            )

            return img_enhanced

        except Exception as e:
            log.error("preprocessing_batch_failed", error=str(e))
            raise e

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Helper to move GPU tensors back to CPU Numpy arrays for storage/saving.
        
        Args:
            tensor (torch.Tensor): Input tensor (B, C, H, W) range 0-1.
            
        Returns:
            np.ndarray: Array (B, C, H, W) range 0-255, dtype uint8.
        """
        # Clamp to ensure safe casting
        tensor = torch.clamp(tensor, 0, 1)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    # --- Legacy Wrappers for Single Image Support (Optional/Backward Compatibility) ---
    
    def apply_clahe(self, image_array: np.ndarray) -> np.ndarray:
        """Legacy wrapper for single image CLAHE."""
        tensor_out = self.preprocess_batch(image_array)
        # Return only the image content (H, W)
        return self.tensor_to_numpy(tensor_out)[0, 0]

    def denoise(self, image_array: np.ndarray) -> np.ndarray:
        """Legacy wrapper for single image Denoising."""
        # For legacy calls, we just run the full pipeline as it includes denoising
        return self.apply_clahe(image_array)

    def standardize(self, image_array: np.ndarray) -> torch.Tensor:
        """Legacy wrapper for returning tensor."""
        return self.preprocess_batch(image_array)