class FluxError(Exception):
    """Base exception for Flux Platform"""
    pass

class StorageError(FluxError):
    """Raised when S3/MinIO operations fail after retries"""
    pass

class DataIngestionError(FluxError):
    """Raised when DICOM parsing or validation fails"""
    pass