import tritonclient.grpc as grpcclient
import numpy as np
from src.common.logger import configure_logger
from src.common.exceptions import FluxError

log = configure_logger("triton_connector")

class TritonClientWrapper:
    def __init__(self, url="localhost:8001", model_name="flux_segmentation"):
        self.url = url
        self.model_name = model_name
        self.client = None
        self._connect()

    def _connect(self):
        try:
            self.client = grpcclient.InferenceServerClient(url=self.url)
            if not self.client.is_server_live():
                log.warning("triton_server_not_live", url=self.url)
        except Exception as e:
            log.error("triton_connection_fail", error=str(e))

    def infer(self, image_numpy: np.ndarray):
        """
        Sends numpy array to Triton and returns prediction.
        Input: (1, 96, 96) or (1, 1, 96, 96)
        """
        if self.client is None:
            # Mock response for Testbed if server isn't running
            return np.zeros((1, 96, 96))

        # Ensure Input Shape (Batch, C, H, W)
        if image_numpy.ndim == 3:
            input_data = np.expand_dims(image_numpy, axis=0)
        else:
            input_data = image_numpy

        inputs = [
            grpcclient.InferInput("input", input_data.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_data.astype(np.float32))

        outputs = [
            grpcclient.InferRequestedOutput("output")
        ]

        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            return response.as_numpy("output")
        except Exception as e:
            log.error("inference_request_failed", error=str(e))
            raise FluxError(f"Triton Error: {e}")