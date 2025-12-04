import pydicom
import numpy as np
import nibabel as nib  # NEW
import io
import os
import gc # NEW: Garbage collection
from src.common.storage import StorageClient
from src.common.logger import configure_logger
from src.common.exceptions import DataIngestionError
from src.common.config import settings
import structlog
from src.common.utils.security import generate_trace_id

log = configure_logger("ingestion_service")

class DicomProcessor:
    def __init__(self):
        self.storage = StorageClient()

    def deidentify(self, ds: pydicom.dataset.FileDataset) -> pydicom.dataset.FileDataset:
        tags_to_clear = ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientAddress']
        for tag in tags_to_clear:
            if hasattr(ds, tag):
                setattr(ds, tag, "ANONYMOUS")
        return ds

    def process_and_upload(self, dicom_path: str) -> str:

        # 1. OPTIMIZATION: Generate Trace ID
        trace_id = generate_trace_id()
        
        # Bind trace_id to all logs in this context
        request_log = log.bind(trace_id=trace_id)
        request_log.info("processing_start", file=dicom_path)

        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"File not found: {dicom_path}")

        log.info("processing_start", file=dicom_path)
        
        try:
            # 1. Read
            ds = pydicom.dcmread(dicom_path)
            
            # 2. Extract Metadata for Audit
            audit_tags = {
                "original_modality": ds.get("Modality", "UNKNOWN"),
                "ingest_source": "manual_upload",
                "trace_id": trace_id
            }

            # 3. De-identify
            ds = self.deidentify(ds)
            
            # 4. Convert to NIfTI (Standard Medical Format)
            # Create an identity affine (since we don't have full orientation info in this dummy)
            # In real fluoroscopy, we would calculate this from ImageOrientationPatient
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Normalize
            if np.max(pixel_array) > 0:
                pixel_array = pixel_array / np.max(pixel_array)

            # Create NIfTI object
            # Use specific rotation if needed, here we assume identity for simplicity
            affine = np.eye(4) 
            nifti_img = nib.Nifti1Image(pixel_array, affine)

            # 5. Save to Memory Buffer as Compressed .nii.gz
            bio = io.BytesIO()
            nib.save(nifti_img, bio)
            
            # 6. Upload
            object_name = f"{os.path.basename(dicom_path)}.nii.gz"
            s3_path = self.storage.upload_file_object(
                bio, 
                settings.MINIO_BUCKET_PROCESSED, 
                object_name, 
                metadata=audit_tags
            )
            
            # Optimization: Explicitly free memory for large arrays
            del ds, pixel_array, nifti_img, bio
            gc.collect()

            log.info("processing_complete", s3_path=s3_path)
            return object_name

        except Exception as e:
            log.error("processing_error", error=str(e), file=dicom_path)
            raise DataIngestionError(str(e))