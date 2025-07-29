#!/usr/bin/env python3
"""Test script to verify disk-based interpolation tensor size fix."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add the project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from rife_app.utils.disk_based_interpolation import disk_based_interpolate
from rife_app.utils.framing import pad_tensor_for_rife
from rife_app.models.loader import load_rife_model

def create_test_frames(width: int = 2160, height: int = 1080):
    """Create test frames with specific dimensions that need padding."""
    # Create simple gradient images
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradients for visual distinction
    for i in range(height):
        frame1[i, :, 0] = int(255 * i / height)  # Red gradient
        frame2[i, :, 1] = int(255 * i / height)  # Green gradient
    
    # Convert to tensors
    tensor1 = torch.from_numpy(frame1.transpose(2, 0, 1)).float() / 255.0
    tensor2 = torch.from_numpy(frame2.transpose(2, 0, 1)).float() / 255.0
    
    # Add batch dimension
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)
    
    return tensor1, tensor2

def test_disk_interpolation():
    """Test disk-based interpolation with padding."""
    print("🧪 Testing disk-based interpolation fix...")
    
    # Create test frames with dimensions that need padding
    print("\n1. Creating test frames (2160x1080)...")
    frame1, frame2 = create_test_frames(2160, 1080)
    print(f"   Original frame shape: {frame1.shape}")
    
    # Apply padding
    print("\n2. Applying padding for RIFE...")
    padded_frame1, padding_info1 = pad_tensor_for_rife(frame1)
    padded_frame2, padding_info2 = pad_tensor_for_rife(frame2)
    print(f"   Padded frame shape: {padded_frame1.shape}")
    print(f"   Padding info: {padding_info1}")
    
    # Load model
    print("\n3. Loading RIFE model...")
    model = load_rife_model(model_name="RIFE", scale=1.0)
    
    # Test disk-based interpolation
    print("\n4. Running disk-based interpolation...")
    try:
        video_path, status = disk_based_interpolate(
            padded_frame1,
            padded_frame2,
            model,
            target_frames=5,
            device="cuda" if torch.cuda.is_available() else "cpu",
            original_size=padding_info1  # Pass padding info for final cropping
        )
        
        print(f"\n✅ Success! {status}")
        if video_path:
            print(f"   Output video: {video_path}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n🎯 Test complete!")

if __name__ == "__main__":
    test_disk_interpolation()