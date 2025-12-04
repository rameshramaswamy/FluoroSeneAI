import pytest
import numpy as np
from src.ingestion.quality import QualityGate

def test_quality_gate_dark():
    gate = QualityGate()
    # Create black image
    dark_img = np.zeros((100, 100), dtype=np.uint8)
    
    report = gate.evaluate(dark_img)
    assert report.is_valid == False
    assert report.reason == "REJECTED_DARK"
    assert report.metrics['intensity'] == 0.0

def test_quality_gate_valid():
    gate = QualityGate()
    # Create Noise Image (Simulating Texture/Content)
    valid_img = np.random.randint(50, 255, (100, 100), dtype=np.uint8)
    
    report = gate.evaluate(valid_img)
    assert report.is_valid == True
    assert report.reason == "VALID"