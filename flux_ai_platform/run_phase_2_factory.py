import sys
import os
import numpy as np
import nibabel as nib
import cv2
import tempfile
import concurrent.futures
from tqdm import tqdm
from unittest.mock import MagicMock

# ... imports ...
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.ingestion.processor import DicomProcessor
from src.ingestion.quality import QualityGate
from src.preprocessing.enhancement import ImageEnhancer
from src.active_learning.suggester import AnnotationSuggester
from src.active_learning.cvat_manager import CvatManager
from src.common.storage import StorageClient
from src.common.logger import configure_logger
from src.common.config import settings
from src.ingestion.deduplication import Deduplicator

log = configure_logger("optimized_factory")

class SmartPipeline:
    def __init__(self, batch_size=4, uncertainty_threshold=0.1):
        self.batch_size = batch_size
        self.uncertainty_threshold = uncertainty_threshold
        
        # Services
        self.storage = StorageClient()
        self.ingestor = DicomProcessor()
        self.quality_gate = QualityGate()
        self.deduplicator = Deduplicator() # NEW
        self.enhancer = ImageEnhancer(use_gpu=True)
        self.suggester = AnnotationSuggester(model_path="flux_model_v0.1.pt")
        self.io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.cvat = CvatManager()
        
        # Mock CVAT
        try: self.cvat.connect()
        except: 
            self.cvat.client = MagicMock()
            self.cvat.client.tasks.create_from_data.return_value.id = 999

    def _io_upload_task(self, file_name, enhanced_img, mask_img):
        """Async task to write temp files and upload to CVAT"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img, \
                 tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
                
                # Write to disk
                cv2.imwrite(tmp_img.name, enhanced_img[0]) # (1, H, W) -> (H, W)
                cv2.imwrite(tmp_mask.name, mask_img * 255)
                
                # Network I/O
                task_id = self.cvat.create_task(
                    task_name=f"Auto_{file_name}",
                    labels=["instrument", "vessel"],
                    image_path=tmp_img.name,
                    annotation_path=tmp_mask.name
                )
                
                os.unlink(tmp_img.name)
                os.unlink(tmp_mask.name)
                return task_id
        except Exception as e:
            log.error("async_upload_failed", error=str(e), file=file_name)
            return None

    def run_batch(self, file_paths):
        log.info("batch_start", total=len(file_paths))
        
        valid_files = []
        raw_arrays = []

        # --- STAGE 1: CPU Filters (Deduplication + Quality) ---
        for f_path in file_paths:
            try:
                import pydicom
                ds = pydicom.dcmread(f_path)
                
                # 1. Deduplication Check (Fastest)
                if self.deduplicator.is_duplicate(ds.pixel_array):
                    log.info("dropped_duplicate", file=os.path.basename(f_path))
                    continue

                # 2. Quality Check
                report = self.quality_gate.evaluate(ds.pixel_array)
                if not report.is_valid:
                    continue

                # Prepare for GPU
                valid_files.append(f_path)
                arr = ds.pixel_array.astype(float)
                arr = (arr / np.max(arr) * 255).astype(np.uint8)
                raw_arrays.append(arr)

            except Exception: continue

        if not valid_files: return

        # --- STAGE 2: GPU Processing ---
        batch_np = np.stack(raw_arrays)
        enhanced_tensor = self.enhancer.preprocess_batch(batch_np)
        
        # 3. Inference with Uncertainty
        masks_np, uncertainties = self.suggester.predict_batch_with_uncertainty(enhanced_tensor)
        enhanced_np = self.enhancer.tensor_to_numpy(enhanced_tensor)

        # --- STAGE 3: Active Learning Filter ---
        futures = []
        for i, f_path in enumerate(valid_files):
            score = uncertainties[i]
            fname = os.path.basename(f_path)

            # Smart Filter: Only annotate if model is confused
            if score < self.uncertainty_threshold:
                log.info("dropped_confident_prediction", file=fname, uncertainty=f"{score:.4f}")
                # Optional: Save these as "Silver Standard" data (auto-labeled) without human review
                continue
            
            log.info("uploading_uncertain_case", file=fname, uncertainty=f"{score:.4f}")
            
            future = self.io_pool.submit(
                self._io_upload_task, 
                fname, 
                enhanced_np[i], 
                masks_np[i]
            )
            futures.append(future)

# --- Test Data Generator (Same as before) ---
def generate_batch_data(count=10):
    files = []
    # Generate mix of Good and Bad
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import UID
    
    os.makedirs("tests/data/batch_load", exist_ok=True)
    
    for i in range(count):
        is_bad = i % 5 == 0 # Every 5th is bad
        fname = f"tests/data/batch_load/img_{i:03d}.dcm"
        
        ds = FileDataset(fname, {}, preamble=b"\0"*128)
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = UID('1.2.840.10008.1.2.1')
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        img = np.zeros((256, 256), dtype=np.uint8)
        if not is_bad:
            cv2.circle(img, (128, 128), 50, 200, -1)
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        ds.PixelData = img.tobytes()
        ds.Rows, ds.Columns = 256, 256
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.save_as(fname)
        files.append(fname)
    return files

def main():
    # Generate 12 files to test batching (Batch size 4 -> 3 batches)
    files = generate_batch_data(12)
    
    pipeline = OptimizedPipeline(batch_size=4)
    
    # Process in chunks
    chunk_size = 4
    for i in tqdm(range(0, len(files), chunk_size), desc="Processing Batches"):
        batch_files = files[i : i + chunk_size]
        pipeline.run_batch(batch_files)
        
    log.info("pipeline_finished_all_batches")

if __name__ == "__main__":
    main()