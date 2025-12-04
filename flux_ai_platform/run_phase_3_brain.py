import sys
import os
import shutil
import numpy as np
import nibabel as nib
import torch

sys.path.append(os.path.join(os.getcwd(), "src"))

from src.training.tuning.optimizer import HyperparamOptimizer
from src.training.data_loader import FluxDataLoader
from src.common.logger import configure_logger

log = configure_logger("phase_3_final_runner")

def generate_phase_2_output(data_dir="experiments/data_cache"):
    """
    Simulates the output of Phase 2 (Data Factory) so Phase 3 has files to read.
    Creates 20 fake NIfTI files.
    """
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    
    log.info("generating_synthetic_nifti_data", dir=data_dir)
    
    for i in range(20):
        # Simulate Patient Groups (Patient_0, Patient_1, etc.)
        patient_id = f"Patient_{i // 4}" # 4 frames per patient
        
        # Create random image (H, W) -> (96, 96) matches config
        img_data = np.random.rand(96, 96).astype(np.float32)
        
        # Create random mask (0 or 1)
        mask_data = np.random.randint(0, 2, (96, 96)).astype(np.float32)
        
        # Save as NIfTI
        img_nii = nib.Nifti1Image(img_data, np.eye(4))
        mask_nii = nib.Nifti1Image(mask_data, np.eye(4))
        
        # Filename convention used by DataLoader
        nib.save(img_nii, f"{data_dir}/enhanced_{patient_id}_{i}.nii.gz")
        nib.save(mask_nii, f"{data_dir}/suggestion_{patient_id}_{i}.nii.gz")

    return data_dir

def main():
    log.info("pipeline_start", phase="3.1", status="integration_test")

    # 1. Prepare Data (Simulate Phase 2 Output)
    data_dir = generate_phase_2_output()

    # 2. Initialize Smart Data Loader
    # This automatically splits by Patient ID to prevent leakage
    loader_factory = FluxDataLoader(data_dir=data_dir, batch_size=4)
    train_loader, val_loader = loader_factory.get_loaders(n_splits=3, fold_index=0)

    # 3. Run Hyperparameter Optimization
    # Now running on actual NIfTI files loaded via MONAI
    log.info("starting_optimization")
    optimizer = HyperparamOptimizer(n_trials=2) # Keep low for demo
    best_params = optimizer.run_optimization(train_loader, val_loader)

    print("\n🏆 CHAMPION CONFIGURATION (Trained on NIfTI):")
    print(best_params)

if __name__ == "__main__":
    main()