import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandRotate90d, RandFlipd, RandZoomd, EnsureTyped
)
from src.common.logger import configure_logger
from src.common.config import settings

log = configure_logger("flux_data_loader")

class FluxDataLoader:
    def __init__(self, data_dir: str, batch_size: int = 4, cache_rate: float = 1.0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        
        # Define Transformations
        # 1. Load NIfTI
        # 2. Add Channel Dim
        # 3. Normalize Intensity (0-1)
        # 4. Convert to Tensor
        self.base_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
        ]
        
        # Augmentations (Only for Training)
        self.train_transforms = Compose(self.base_transforms + [
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
        ])
        
        self.val_transforms = Compose(self.base_transforms)

    def _parse_dataset(self):
        """
        Scans directory for matching image/mask pairs.
        Expected structure from Phase 2:
          - enhanced_patient123_frame01.nii.gz
          - suggestion_patient123_frame01.nii.gz (Treating 'suggestion' as ground truth for now)
        """
        # Look for processed files from Phase 2
        images = sorted(glob.glob(os.path.join(self.data_dir, "enhanced_*.nii.gz")))
        masks = sorted(glob.glob(os.path.join(self.data_dir, "suggestion_*.nii.gz")))
        
        if len(images) != len(masks) or len(images) == 0:
            log.warning("dataset_mismatch_or_empty", images=len(images), masks=len(masks))
            return []

        data_dicts = []
        patient_ids = []

        for img_path, mask_path in zip(images, masks):
            # Extract Patient ID for Group Splitting
            # Filename format: enhanced_patientID_frame.nii.gz
            # Simple heuristic: use the filename itself as group if no ID found
            filename = os.path.basename(img_path)
            try:
                # Mock parsing logic: assume "patientID" is part of string
                # Real implementation would use regex
                patient_id = filename.split("_")[1] 
            except IndexError:
                patient_id = filename

            data_dicts.append({"image": img_path, "label": mask_path, "patient_id": patient_id})
            patient_ids.append(patient_id)
            
        return data_dicts, patient_ids

    def get_loaders(self, n_splits=5, fold_index=0):
        """
        Returns Train and Val DataLoaders split by Patient ID (GroupKFold).
        """
        data_dicts, patient_ids = self._parse_dataset()
        
        if not data_dicts:
            raise ValueError("No data found! Did you run Phase 2?")

        # Group K-Fold to prevent leakage
        gkf = GroupKFold(n_splits=n_splits)
        
        # Dummy X and y, we only care about indices and groups
        splits = list(gkf.split(data_dicts, y=patient_ids, groups=patient_ids))
        train_idx, val_idx = splits[fold_index]
        
        train_files = [data_dicts[i] for i in train_idx]
        val_files = [data_dicts[i] for i in val_idx]
        
        log.info("data_split_created", 
                 train_size=len(train_files), 
                 val_size=len(val_files), 
                 fold=fold_index)

        # Create MONAI CacheDatasets
        # CacheRate=1.0 loads entire dataset into RAM for speed
        train_ds = CacheDataset(data=train_files, transform=self.train_transforms, cache_rate=self.cache_rate)
        val_ds = CacheDataset(data=val_files, transform=self.val_transforms, cache_rate=self.cache_rate)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=1)
        
        return train_loader, val_loader