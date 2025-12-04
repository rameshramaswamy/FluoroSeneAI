import sys
import os
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

# Set Environment Variables for Local Dev (Mocking k8s secrets)
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MINIO_ACCESS_KEY"] = "fluxadmin"
os.environ["MINIO_SECRET_KEY"] = "fluxpassword123"

sys.path.append(os.path.join(os.getcwd(), "src"))

from src.ingestion.processor import DicomProcessor
from src.training.trainer import FluxTrainer
from src.common.logger import configure_logger

log = configure_logger("pipeline_runner")
from src.training.config_schema import UNetConfig # Import Schema

def main():
    log.info("pipeline_start", phase="1", status="initializing")

    # 1. Generate Data
    dicom_file = create_synthetic_dicom("tests/data/synthetic.dcm")

    # 2. Ingestion (Now with Trace IDs and Hashing)
    try:
        ingestor = DicomProcessor()
        tensor_key = ingestor.process_and_upload(dicom_file)
    except Exception as e:
        log.critical("pipeline_fail_ingestion", error=str(e))
        return

    # 3. Training (Now with Explicit Configuration)
    try:
        # Define specific architecture for Fluoroscopy
        fluoro_config = UNetConfig(
            channels=(32, 64, 128, 256, 512), # Deeper network
            num_res_units=3
        )
        
        trainer = FluxTrainer(config=fluoro_config)
        artifact = trainer.run_dummy_pipeline(tensor_key)
        
        log.info("pipeline_success", artifact=artifact, config=fluoro_config.model_dump())
    except Exception as e:
        log.critical("pipeline_fail_training", error=str(e))
        return

if __name__ == "__main__":
    main()