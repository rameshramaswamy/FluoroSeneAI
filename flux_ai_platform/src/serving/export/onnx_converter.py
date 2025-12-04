import torch
import onnx
import os
from src.training.models.factory import ModelFactory
from src.training.config_schema import TrainingConfig
from src.common.logger import configure_logger

log = configure_logger("onnx_exporter")

class OnnxExporter:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cpu") # Export is usually safer on CPU

    def export(self, weights_path: str, output_path: str):
        """
        Converts PyTorch checkpoint to ONNX graph.
        """
        try:
            # 1. Load Model Structure
            model = ModelFactory.create_model(self.config).to(self.device)
            
            # 2. Load Weights
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
                log.info("weights_loaded", path=weights_path)
            else:
                log.warning("weights_not_found_using_random", path=weights_path)

            model.eval()

            # 3. Create Dummy Input (Batch, Channel, H, W)
            # Must match the input dimensions expected by the model
            dummy_input = torch.randn(
                1, 1, 
                self.config.roi_size[0], 
                self.config.roi_size[1], 
                requires_grad=True
            ).to(self.device)

            # 4. Export
            # Dynamic axes allow the model to accept different batch sizes (Critical for Triton)
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 5. Verify
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            log.info("export_success", output=output_path)
            return output_path

        except Exception as e:
            log.error("export_failed", error=str(e))
            raise e