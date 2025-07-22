import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.nn import functional as F
from typing import Tuple, Optional

def get_video_info(video_path: Path) -> Optional[dict]:
    """Reads video file and returns properties."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        info = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

def pad_tensor_for_rife(tensor: torch.Tensor, multiple: int = 32, min_size: int = 512) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pads a tensor to dimensions that are a multiple of a given number, with a minimum size."""
    _n, _c, h, w = tensor.shape
    
    # Calculate the target padded height and width
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple

    # Enforce minimum size
    ph = max(min_size, ph)
    pw = max(min_size, pw)

    padding = (0, pw - w, 0, ph - h)
    return F.pad(tensor, padding), (h, w)

def save_tensor_as_image(tensor: torch.Tensor, path: Path, original_size: Tuple[int, int]):
    """Crops a tensor to original size and saves it as an image file.
    
    NOTE: This function now properly handles RGB tensors and converts to BGR only for cv2.imwrite.
    """
    h_orig, w_orig = original_size
    # Select the image from the batch, then detach, move to CPU, convert to numpy, and transpose axes
    img_to_save_permuted = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    img_to_save_cropped = img_to_save_permuted[:h_orig, :w_orig, :] 
    img_to_save_uint8 = (img_to_save_cropped * 255).clip(0, 255).astype(np.uint8)
    # Convert RGB to BGR for cv2.imwrite
    img_to_save_bgr = cv2.cvtColor(img_to_save_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_to_save_bgr) 