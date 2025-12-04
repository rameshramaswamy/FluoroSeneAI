import os

def generate_triton_config(model_name: str, max_batch_size: int = 8):
    """
    Generates the Protocol Buffer text config for Triton.
    Enables:
    1. Dynamic Batching (collects requests for 5ms to form a batch)
    2. TensorRT Optimization (FP16)
    """
    config = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, 96, 96 ] 
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ 2, 96, 96 ]
  }}
]

# 1. Dynamic Batching
# Waits up to 3000 microseconds (3ms) to fill a batch of {max_batch_size}.
# This increases throughput dramatically for video streams.
dynamic_batching {{
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 3000
}}

# 2. Hardware Acceleration
# Automatically converts ONNX to TensorRT plan on startup
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [ {{
      name : "tensorrt"
      parameters {{ key: "precision_mode" value: "FP16" }}
      parameters {{ key: "max_workspace_size_bytes" value: "1073741824" }}
    }}]
  }}
}}

# 3. Instance Group
# Run 1 instance on GPU 0
instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""
    return config

def setup_model_repository(repo_path: str, model_name: str, onnx_path: str):
    """
    Sets up the strict folder structure required by Triton:
    /model_repository
      /my_model
        /1
          model.onnx
        config.pbtxt
    """
    model_dir = os.path.join(repo_path, model_name)
    version_dir = os.path.join(model_dir, "1")
    
    os.makedirs(version_dir, exist_ok=True)
    
    # Move/Copy ONNX file
    import shutil
    shutil.copy(onnx_path, os.path.join(version_dir, "model.onnx"))
    
    # Write Config
    config_content = generate_triton_config(model_name)
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
        f.write(config_content)
        
    return model_dir