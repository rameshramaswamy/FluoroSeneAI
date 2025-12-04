import cv2
import numpy as np
from dataclasses import dataclass
from src.common.logger import configure_logger
from src.common.config import settings

log = configure_logger("quality_gate")

@dataclass
class QualityReport:
    is_valid: bool
    reason: str = "OK"
    metrics: dict = None

class QualityGate:
    def __init__(self):
        self.blur_threshold = settings.QUALITY_BLUR_THRESHOLD
        self.dark_threshold = settings.QUALITY_DARK_THRESHOLD

    def evaluate(self, image_array: np.ndarray) -> QualityReport:
        """
        Evaluates frame quality and returns a detailed report.
        """
        metrics = {}
        
        # Ensure grayscale
        if image_array.ndim == 3:
            img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            img = image_array

        # 1. Check Brightness (Beam Off Detection)
        mean_intensity = np.mean(img)
        metrics['intensity'] = float(mean_intensity)
        
        if mean_intensity < self.dark_threshold:
            log.warning("frame_rejected", reason="dark", value=mean_intensity)
            return QualityReport(False, "REJECTED_DARK", metrics)

        # 2. Check Blur (Laplacian Variance)
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        metrics['blur_variance'] = float(variance)
        
        if variance < self.blur_threshold:
            log.warning("frame_rejected", reason="blur", value=variance)
            return QualityReport(False, "REJECTED_BLUR", metrics)

        return QualityReport(True, "VALID", metrics)