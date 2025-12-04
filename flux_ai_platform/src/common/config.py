import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MinIO
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "fluxadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "fluxpassword123")
    MINIO_BUCKET_RAW: str = "flux-raw-dicom"
    MINIO_BUCKET_PROCESSED: str = "flux-processed-tensors"
    MINIO_SECURE: bool = False

    # CVAT (Annotation)
    CVAT_HOST: str = os.getenv("CVAT_HOST", "http://localhost:8080")
    CVAT_USER: str = os.getenv("CVAT_USER", "admin")
    CVAT_PASSWORD: str = os.getenv("CVAT_PASSWORD", "password")

    # Quality Gate Defaults
    QUALITY_BLUR_THRESHOLD: float = 100.0
    QUALITY_DARK_THRESHOLD: float = 10.0
    KEYCLOAK_URL: str = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
    KEYCLOAK_REALM: str = os.getenv("KEYCLOAK_REALM", "flux-realm")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "flux_dev_secret") # Fallback
    
    class Config:
        env_file = ".env"

settings = Settings()