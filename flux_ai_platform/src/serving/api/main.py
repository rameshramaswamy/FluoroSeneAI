from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
import numpy as np
import cv2
import time
import asyncio
from src.serving.client.triton_client import TritonClientWrapper
from src.common.logger import configure_logger

log = configure_logger("api_gateway")
app = FastAPI(title="Flux AI Gateway", version="1.1")

# Initialize Client
triton_client = TritonClientWrapper(url="triton:8001")

def process_frame(image_bytes: bytes):
    """
    CPU-bound task: Decoding and Preprocessing.
    We isolate this function to run it in a separate thread pool 
    so it doesn't block the async event loop.
    """
    # 1. Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 2. Resize to 96x96 (Model Spec)
    img_resized = cv2.resize(img, (96, 96))
    
    # 3. Normalize & Shape (1, 1, 96, 96)
    return (img_resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]

@app.websocket("/ws/segment")
async def websocket_endpoint(websocket: WebSocket):
    """
    High-Performance Streaming Endpoint.
    Protocol: Client sends Binary Image -> Server sends JSON Mask
    """
    await websocket.accept()
    log.info("websocket_connected", client=websocket.client.host)
    
    try:
        while True:
            # 1. Receive Raw Bytes (Lowest overhead)
            data = await websocket.receive_bytes()
            start = time.perf_counter()

            # 2. Offload CPU work (Decoding) to ThreadPool
            # This is critical for maintaining high throughput
            input_tensor = await run_in_threadpool(process_frame, data)
            
            if input_tensor is None:
                await websocket.send_json({"error": "Invalid Image"})
                continue

            # 3. Inference (gRPC is inherently I/O bound, so awaitable if wrapper supports it)
            # For now, our wrapper is sync, so we wrap it too or assume low latency
            prediction = await run_in_threadpool(triton_client.infer, input_tensor)
            
            # 4. Post-process (Argmax)
            mask = np.argmax(prediction, axis=1)[0].astype(np.uint8)
            
            # 5. Serialize Response
            # In production, we might send back binary bytes for the mask
            # For now, we send metadata + latency
            latency = (time.perf_counter() - start) * 1000
            
            await websocket.send_json({
                "shape": mask.shape,
                "latency_ms": f"{latency:.2f}",
                # "mask_b64": ... (Optional: encode mask here)
            })

    except WebSocketDisconnect:
        log.info("websocket_disconnected")
    except Exception as e:
        log.error("websocket_error", error=str(e))
        try:
            await websocket.close()
        except: pass

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "streaming_ready"}