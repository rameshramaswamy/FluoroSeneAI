import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ingestion.processor import DicomProcessor

# Mock the Storage Client so we don't need real MinIO for unit tests
@patch("src.ingestion.processor.StorageClient")
def test_ingestion_logic(MockStorage):
    # Setup
    mock_storage_instance = MockStorage.return_value
    mock_storage_instance.upload_numpy.return_value = "s3://bucket/test.npy"
    
    processor = DicomProcessor()
    
    # Create a dummy DICOM dataset object
    class DummyDS:
        PixelData = b'\x00\x01'
        Rows = 10
        Columns = 10
        pixel_array = np.zeros((10, 10))
        PatientName = "John Doe"
        
    # Test De-identification
    ds = DummyDS()
    ds = processor.deidentify(ds) # type: ignore
    assert ds.PatientName == "ANONYMOUS"

    # Test Validation (Should fail if PixelData missing)
    class BadDS:
        PatientID = "123"
        
    with pytest.raises(Exception):
        processor.validate_dicom(BadDS()) # type: ignore