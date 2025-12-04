import imagehash
from PIL import Image
import numpy as np
from src.common.logger import configure_logger

log = configure_logger("deduplication_service")

class Deduplicator:
    def __init__(self, similarity_threshold=5):
        # In production, this "seen_hashes" would be a Redis Set or Bloom Filter
        self.seen_hashes = set()
        self.similarity_threshold = similarity_threshold

    def is_duplicate(self, image_array: np.ndarray) -> bool:
        """
        Returns True if a perceptually similar image has already been processed.
        """
        try:
            # Convert Numpy -> PIL for imagehash
            if image_array.ndim == 3 and image_array.shape[0] in [1, 3]:
                # Handle (C, H, W) -> (H, W, C) or (H, W)
                if image_array.shape[0] == 1:
                    img_pil = Image.fromarray(image_array[0].astype('uint8'))
                else:
                    img_pil = Image.fromarray(np.transpose(image_array, (1, 2, 0)).astype('uint8'))
            else:
                img_pil = Image.fromarray(image_array.astype('uint8'))

            # Calculate Perceptual Hash (Robust to slight noise/gamma changes)
            current_hash = imagehash.phash(img_pil)
            
            # Check against cache
            # Note: Set lookup is O(1) for exact matches. 
            # For "Near" matches (Hamming distance), we ideally use a VP-Tree.
            # Here we do exact string matching of the hash for speed.
            if current_hash in self.seen_hashes:
                log.info("duplicate_detected", hash=str(current_hash))
                return True
            
            self.seen_hashes.add(current_hash)
            return False

        except Exception as e:
            log.error("deduplication_error", error=str(e))
            return False # Default to processing if check fails