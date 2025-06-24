import torch
from typing import List

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
    Recursively generates n intermediate frames between I0 and I1.
    This is used for the video FPS interpolation tab.
    n_exp: Exponent for 2**n_exp-1 total intermediate frames.
    """
    if n_exp == 0:
        return []
    
    middle = model.inference(I0, I1, scale=model_scale_factor)
    
    if n_exp == 1:
        return [middle]

    first_half = recursive_interpolate_video_frames(I0, middle, n_exp - 1, model, model_scale_factor)
    second_half = recursive_interpolate_video_frames(middle, I1, n_exp - 1, model, model_scale_factor)
    
    return [*first_half, middle, *second_half] 