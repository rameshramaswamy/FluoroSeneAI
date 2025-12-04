import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from cvat_sdk.core.client import Client
from cvat_sdk.core.helpers import ResourceType
from src.common.logger import configure_logger
from src.common.exceptions import FluxError
from src.common.config import settings

log = configure_logger("cvat_manager")

class CvatManager:
    def __init__(self):
        self.host = settings.CVAT_HOST
        self.user = settings.CVAT_USER
        self.password = settings.CVAT_PASSWORD
        self.client = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def connect(self):
        """Authenticates with CVAT with automatic retries."""
        try:
            self.client = Client(self.host)
            self.client.login((self.user, self.password))
            log.info("cvat_connected", host=self.host, user=self.user)
        except Exception as e:
            log.error("cvat_connection_failed", error=str(e))
            raise FluxError(f"Could not connect to CVAT: {e}")

    def ensure_connection(self):
        if not self.client:
            self.connect()

    def create_task(self, task_name: str, labels: list, image_path: str, annotation_path: str = None):
        self.ensure_connection()

        try:
            log.info("creating_cvat_task", name=task_name)
            
            spec = {
                "name": task_name,
                "labels": [{"name": label} for label in labels],
                "project_id": None # Can be parameterized if projects exist
            }
            
            task = self.client.tasks.create_from_data(
                spec=spec,
                resource_type=ResourceType.LOCAL,
                resources=[image_path],
                annotation_path=annotation_path,
                annotation_format="Mask 1.1"
            )
            
            log.info("cvat_task_created", task_id=task.id)
            return task.id

        except Exception as e:
            log.error("cvat_task_creation_error", error=str(e))
            raise FluxError(f"Failed to create CVAT task: {e}")