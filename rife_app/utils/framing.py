import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.nn import functional as F
from typing import Tuple, Optional

def get_video_info(video_path: Path) -> Optional[dict]:
    """Reads video file and returns enhanced properties with temporal analysis."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Enhanced temporal information
        duration = frame_count / fps if fps > 0 else 0
        
        # Video format detection for encoding optimization
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_fourcc = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
        
        info = {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "codec": codec_fourcc,
            "aspect_ratio": width / height if height > 0 else 1.0
        }
        cap.release()
        return info
    except Exception:
        return None

def extract_frames(video_path: Path, start_frame: int, end_frame: int) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Extracts start and end frames from a video and returns them as PIL images.
    
    NOTE: This function now properly handles color space conversion to avoid color shifts.
    OpenCV reads in BGR, we convert once to RGB for PIL, and maintain RGB throughout.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    pil_start_frame, pil_end_frame = None, None

    # Extract start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    ret_start, frame_start_bgr = cap.read()
    if ret_start:
        # Convert BGR to RGB once for PIL Image
        frame_start_rgb = cv2.cvtColor(frame_start_bgr, cv2.COLOR_BGR2RGB)
        pil_start_frame = Image.fromarray(frame_start_rgb)

    # Extract end frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame - 1)
    ret_end, frame_end_bgr = cap.read()
    if ret_end:
        # Convert BGR to RGB once for PIL Image
        frame_end_rgb = cv2.cvtColor(frame_end_bgr, cv2.COLOR_BGR2RGB)
        pil_end_frame = Image.fromarray(frame_end_rgb)
    
    cap.release()
    return pil_start_frame, pil_end_frame

def validate_temporal_alignment(video_paths: list, target_fps: float) -> dict:
    """
    Validates temporal alignment across multiple videos for concatenation.
    
    Args:
        video_paths: List of video file paths
        target_fps: Target FPS for final output
    
    Returns:
        dict: Alignment analysis with recommendations
    """
    alignment_info = {
        "videos": [],
        "fps_consistent": True,
        "resolution_consistent": True,
        "recommendations": []
    }
    
    reference_fps = None
    reference_resolution = None
    
    for i, video_path in enumerate(video_paths):
        info = get_video_info(Path(video_path))
        if not info:
            alignment_info["recommendations"].append(f"Cannot read video {i+1}: {Path(video_path).name}")
            continue
        
        video_analysis = {
            "path": str(video_path),
            "fps": info["fps"],
            "resolution": (info["width"], info["height"]),
            "duration": info["duration"],
            "frame_count": info["frame_count"],
            "needs_fps_conversion": False,
            "needs_resolution_conversion": False
        }
        
        # FPS consistency check
        if reference_fps is None:
            reference_fps = info["fps"]
        elif abs(info["fps"] - reference_fps) > 0.1:  # 0.1 FPS tolerance
            alignment_info["fps_consistent"] = False
            video_analysis["needs_fps_conversion"] = True
        
        # Resolution consistency check
        current_resolution = (info["width"], info["height"])
        if reference_resolution is None:
            reference_resolution = current_resolution
        elif current_resolution != reference_resolution:
            alignment_info["resolution_consistent"] = False
            video_analysis["needs_resolution_conversion"] = True
        
        alignment_info["videos"].append(video_analysis)
    
    # Generate recommendations
    if not alignment_info["fps_consistent"]:
        alignment_info["recommendations"].append(f"FPS standardization needed (target: {target_fps} fps)")
    
    if not alignment_info["resolution_consistent"]:
        alignment_info["recommendations"].append(f"Resolution standardization needed (reference: {reference_resolution})")
    
    return alignment_info

def pil_to_tensor(img: Image.Image, device) -> torch.Tensor:
    """Converts a PIL Image to a PyTorch tensor.
    
    NOTE: This function now maintains RGB format throughout to avoid color shifts.
    The tensor will contain RGB channels, not BGR.
    """
    # PIL Image is already in RGB format, so we keep it as RGB
    img_rgb_array = np.array(img)
    # Convert HWC to CHW format for PyTorch
    img_tensor_chw = torch.from_numpy(img_rgb_array.transpose(2, 0, 1)).float().to(device) / 255.
    return img_tensor_chw.unsqueeze(0)

def pad_tensor_for_rife(tensor: torch.Tensor, multiple: int = 32, min_size: int = 512, center_padding: bool = True) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Pads a tensor to dimensions that are a multiple of a given number, with a minimum size.
    
    Args:
        tensor: Input tensor to pad
        multiple: Padding multiple (default 32 for RIFE)
        min_size: Minimum tensor size (default 512)
        center_padding: If True, use centered padding; if False, use asymmetric (right/bottom) padding
        
    Returns:
        Tuple of (padded_tensor, (original_h, original_w, pad_top, pad_left))
    """
    _n, _c, h, w = tensor.shape
    
    # Calculate the target padded height and width
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple

    # Enforce minimum size
    ph = max(min_size, ph)
    pw = max(min_size, pw)

    if center_padding:
        # SYSTEMATIC FIX: Use centered padding to match FFmpeg's coordinate system
        # This eliminates the 16-pixel shift by ensuring spatial alignment
        pad_left = (pw - w) // 2
        pad_right = pw - w - pad_left  
        pad_top = (ph - h) // 2
        pad_bottom = ph - h - pad_top
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        
        # Return padding coordinates for precise cropping
        return F.pad(tensor, padding), (h, w, pad_top, pad_left)
    else:
        # Legacy asymmetric padding (right/bottom only) - causes alignment issues
        padding = (0, pw - w, 0, ph - h)
        return F.pad(tensor, padding), (h, w, 0, 0)

def save_tensor_as_image(tensor: torch.Tensor, path: Path, original_size: Tuple[int, int, int, int]):
    """
    Crops a tensor to original size and saves it as an image file.
    
    SYSTEMATIC FIX: Now handles centered padding coordinates for precise cropping.
    This ensures exact spatial alignment with FFmpeg-processed videos.
    
    Args:
        tensor: Padded tensor to crop and save
        path: Output file path
        original_size: Tuple of (original_h, original_w, pad_top, pad_left)
    """
    if len(original_size) == 2:
        # Legacy format compatibility: (h_orig, w_orig)
        h_orig, w_orig = original_size
        pad_top, pad_left = 0, 0
    else:
        # New centered padding format: (h_orig, w_orig, pad_top, pad_left)
        h_orig, w_orig, pad_top, pad_left = original_size
    
    # Select the image from the batch, then detach, move to CPU, convert to numpy, and transpose axes
    img_to_save_permuted = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    
    # SYSTEMATIC FIX: Use precise padding coordinates for cropping
    # This eliminates spatial misalignment by cropping from exact original position
    img_to_save_cropped = img_to_save_permuted[pad_top:pad_top+h_orig, pad_left:pad_left+w_orig, :] 
    
    img_to_save_uint8 = (img_to_save_cropped * 255).clip(0, 255).astype(np.uint8)
    # Convert RGB to BGR for cv2.imwrite
    img_to_save_bgr = cv2.cvtColor(img_to_save_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_to_save_bgr) 