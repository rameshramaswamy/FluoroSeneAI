import sys
import os
import threading
import time
import requests
import uvicorn
import numpy as np
import cv2

# Environment Setup
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.serving.export.onnx_converter import OnnxExporter
from src.serving.triton_config.generator import setup_model_repository
from src.training.config_schema import TrainingConfig
from src.common.logger import configure_logger

log = configure_logger("phase_4_runner")

def start_api_server():
    """Starts FastAPI in a separate thread"""
    from src.serving.api.main import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

def generate_dummy_image():
    """Create a fake PNG image for testing"""
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()

def main():
    log.info("pipeline_start", phase="4", status="deployment_setup")

    # 1. Export Model (PyTorch -> ONNX)
    config = TrainingConfig(model_architecture="unet", roi_size=(96, 96))
    exporter = OnnxExporter(config)
    
    # Create a dummy checkpoint to simulate a trained model
    dummy_ckpt = "dummy_model.pt"
    # We skip creating the actual .pt file and let the exporter init with random weights
    # if the file doesn't exist, as implemented in module 3.
    
    onnx_path = "flux_model.onnx"
    try:
        exporter.export(dummy_ckpt, onnx_path)
    except Exception as e:
        log.critical("export_failed", error=str(e))
        return

    # 2. Setup Triton Repository
    repo_path = os.path.join(os.getcwd(), "model_repository")
    model_dir = setup_model_repository(repo_path, "flux_segmentation", onnx_path)
    log.info("triton_repo_ready", path=model_dir)

    # 3. Start API Gateway
    log.info("starting_api_gateway")
    server_thread = threading.Thread(target=start_api_server, daemon=True)
    server_thread.start()
    
    # Wait for server to boot
    time.sleep(3)

    # 4. Run Latency Test (Client Simulation)
    url = "http://127.0.0.1:8000/v1/segment"
    image_bytes = generate_dummy_image()
    
    log.info("sending_test_request", url=url)
    try:
        # We Mock the Triton Connection inside the API for this testbed
        # because we don't have a real Triton Docker container running.
        response = requests.post(url, files={"file": ("test.png", image_bytes, "image/png")})
        
        if response.status_code == 200:
            data = response.json()
            log.info("test_success", 
                     latency=f"{data['latency_ms']:.2f}ms", 
                     mask_shape=data['mask_shape'])
            
            if data['latency_ms'] < 35:
                print("\n🚀 SUCCESS: Latency is under 35ms (Real-time Ready)!")
            else:
                print("\n⚠️ WARNING: Latency is high. (Note: Without GPU/TensorRT, this is expected).")
        else:
            log.error("test_failed", status=response.status_code, detail=response.text)

    except Exception as e:
        log.error("connection_failed", error=str(e))

if __name__ == "__main__":
    main()