import io
import numpy as np
from minio import Minio
from minio.error import S3Error
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import settings
from .logger import configure_logger
from .exceptions import StorageError
from src.common.utils.security import calculate_sha256

log = configure_logger("storage_client")

class StorageClient:
    def __init__(self):
        try:
            self.client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE
            )
            # Fail fast if connection is bad
            self.client.list_buckets()
        except Exception as e:
            log.critical("failed_to_connect_storage", error=str(e))
            raise StorageError(f"Could not connect to MinIO: {e}")
            
        self._ensure_buckets()

    def _ensure_buckets(self):
        for bucket in [settings.MINIO_BUCKET_RAW, settings.MINIO_BUCKET_PROCESSED]:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                log.info("bucket_created", bucket=bucket)

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(S3Error)
    )
    def upload_file_object(self, file_data: io.BytesIO, bucket: str, object_name: str, metadata: dict = None) -> str:
        """
        Generic upload with Tags for Audit Trails.
        """
        try:
            
            file_view = file_data.getbuffer()
            file_hash = calculate_sha256(file_view)
            tags = Tags(for_object=True)
            tags["sha256"] = file_hash
            
            if metadata:
                for k, v in metadata.items():
                    tags[k] = str(v)

            length = file_data.getbuffer().nbytes
            file_data.seek(0)
            
            self.client.put_object(
                bucket,
                object_name,
                file_data,
                length=length,
                tags=tags # Attach Audit Tags
            )
            log.info("upload_success", object=object_name, size=bio.getbuffer().nbytes)
            return f"s3://{settings.MINIO_BUCKET_PROCESSED}/{object_name}"
        except Exception as e:
            log.error("upload_failed", object=object_name, error=str(e))
            raise S3Error(str(e)) # Trigger retry

    def download_numpy(self, object_name: str) -> np.ndarray:
        response = None
        try:
            response = self.client.get_object(settings.MINIO_BUCKET_PROCESSED, object_name)
            bio = io.BytesIO(response.read())
            return np.load(bio)
        except Exception as e:
            log.error("download_failed", object=object_name, error=str(e))
            raise StorageError(f"Failed to download {object_name}")
        finally:
            if response:
                response.close()