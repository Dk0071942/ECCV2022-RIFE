import torch
from typing import List, Dict

def generate_interpolated_frames(img0_tensor: torch.Tensor, img1_tensor: torch.Tensor, exp_value: int, model) -> List[torch.Tensor]:
    """
    Generates interpolated frames between two tensors.
    This logic is a direct translation from the original inference_img.py script.
    """
    img_list = [img0_tensor, img1_tensor]
    for i in range(exp_value):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1_tensor)
        img_list = tmp
    
    return img_list

def recursive_interpolate_video_frames(I0: torch.Tensor, I1: torch.Tensor, n_exp: int, model, model_scale_factor: float) -> List[torch.Tensor]:
    """
    DEPRECATED: Use progressive_interpolate_frames instead.
    This function is kept for backward compatibility.
    """
    return progressive_interpolate_frames(I0, I1, n_exp, model, model_scale_factor)

def progressive_interpolate_frames(I0: torch.Tensor, I1: torch.Tensor, n_exp: int, model, model_scale_factor: float) -> List[torch.Tensor]:
    """
    Progressive refinement interpolation that maintains timestep=0.5 throughout.
    
    Instead of recursive subdivision, this approach:
    1. First generates the middle frame
    2. Then generates quarter frames (between start-middle and middle-end)
    3. Continues refining by interpolating between adjacent frames
    
    This maintains optimal timestep=0.5 for all interpolations, which is what RIFE
    was trained for, resulting in better quality than arbitrary timestep interpolation.
    
    Args:
        I0: Start frame tensor
        I1: End frame tensor
        n_exp: Exponent for 2**n_exp-1 total intermediate frames
        model: RIFE model instance
        model_scale_factor: Scale factor for the model
        
    Returns:
        List of interpolated frames (excluding start and end frames)
    """
    if n_exp == 0:
        return []
    
    # Stage 1: Generate middle frame
    frames: Dict[float, torch.Tensor] = {0.0: I0, 1.0: I1}
    middle = model.inference(I0, I1, scale=model_scale_factor)
    frames[0.5] = middle
    
    if n_exp == 1:
        return [middle]
    
    # Progressive refinement stages
    for stage in range(2, n_exp + 1):
        # Get sorted frame positions
        positions = sorted(frames.keys())
        new_frames = {}
        
        # Interpolate between each adjacent pair
        for i in range(len(positions) - 1):
            pos_a = positions[i]
            pos_b = positions[i + 1]
            pos_mid = (pos_a + pos_b) / 2
            
            # Skip if we already have this position
            if pos_mid in frames:
                continue
            
            # Interpolate at timestep=0.5 between adjacent frames
            frame_a = frames[pos_a]
            frame_b = frames[pos_b]
            frame_mid = model.inference(frame_a, frame_b, scale=model_scale_factor)
            new_frames[pos_mid] = frame_mid
        
        # Add new frames to our collection
        frames.update(new_frames)
        
        # Check if we have enough frames
        if len(frames) - 2 >= (2**n_exp - 1):
            break
    
    # Return frames in order (excluding start and end)
    result = []
    for pos in sorted(frames.keys()):
        if 0.0 < pos < 1.0:
            result.append(frames[pos])
    
    return result 