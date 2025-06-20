import os
import cv2
import torch
import numpy as np
from torch.nn import functional as F
import shutil
import subprocess
import datetime
from pathlib import Path
import sys
from tqdm import tqdm

from rife_utils import (
    get_rife_model,
    get_video_properties,
    cv2_frame_reader,
    run_ffmpeg_command,
    transfer_audio_ffmpeg,
    pad_image_for_rife,
    make_rife_inference,
)

# Add the submodule's parent directory to sys.path to allow imports from the main project
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

MODEL_DIR = script_dir / 'train_log'

# Global model variable for RIFE is now managed in rife_utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def main_interpolate(
    input_video_path: str,
    output_dir_path: str,
    exp: int = 1,
    use_fp16: bool = False,
    model_inference_scale_factor: float = 1.0,
    output_resolution_scale_factor: float = 1.0,
    target_fps_override: int = None,
):
    model = get_rife_model()
    if use_fp16 and not torch.cuda.is_available():
        raise ValueError("FP16 is selected, but CUDA is not available.")
    
    original_torch_dtype = torch.get_default_dtype()
    operation_dir = None
    
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if use_fp16 and torch.cuda.is_available():
            torch.set_default_dtype(torch.float16)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        operation_dir = output_dir / f"rife_op_{timestamp}"
        operation_dir.mkdir(exist_ok=True)
        
        frames_output_dir = operation_dir / "interpolated_frames"
        frames_output_dir.mkdir(exist_ok=True)

        input_video_path_obj = Path(input_video_path)
        output_video_filename = f"{input_video_path_obj.stem}_interpolated_{timestamp}.mp4"
        final_output_video_path = str(output_dir / output_video_filename)

        input_w, input_h, original_fps, total_frames_input_video = get_video_properties(input_video_path)

        if total_frames_input_video == 0:
            raise ValueError("Input video contains zero frames.")

        output_w = int(input_w * output_resolution_scale_factor)
        output_h = int(input_h * output_resolution_scale_factor)
        output_w = max(1, output_w)
        output_h = max(1, output_h)
        
        if target_fps_override and target_fps_override > 0:
            final_fps = target_fps_override
        else:
            final_fps = original_fps * (2**exp)

        print(f"RIFE Input: {total_frames_input_video} frames, {original_fps:.2f} FPS, {input_w}x{input_h}")
        print(f"RIFE Output: Target FPS: {final_fps:.2f}, Resolution: {output_w}x{output_h}")
        
        videogen = cv2_frame_reader(input_video_path)
        last_frame_np = next(videogen).copy()

        last_frame_tensor = torch.from_numpy(last_frame_np.transpose(2, 0, 1)).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        if output_resolution_scale_factor != 1.0:
            last_frame_tensor = F.interpolate(last_frame_tensor, size=(output_h, output_w), mode='bilinear', align_corners=False)
        I1_padded = pad_image_for_rife(last_frame_tensor, model_inference_scale_factor, use_fp16)
        
        _first_frame_tensor_slice = I1_padded[0, :, :output_h, :output_w]
        first_frame_to_save_np = (_first_frame_tensor_slice.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite(str(frames_output_dir / f"frame_{0:07d}.png"), cv2.cvtColor(first_frame_to_save_np, cv2.COLOR_RGB2BGR))
        saved_frame_count = 1
        
        progress_bar = tqdm(total=total_frames_input_video - 1, desc="RIFE Interpolating Frames")

        for _, current_frame_np_orig in enumerate(videogen, start=1):
            current_frame_np = current_frame_np_orig.copy()
            I0_padded = I1_padded

            current_frame_tensor = torch.from_numpy(current_frame_np.transpose(2, 0, 1)).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            if output_resolution_scale_factor != 1.0:
                current_frame_tensor = F.interpolate(current_frame_tensor, size=(output_h, output_w), mode='bilinear', align_corners=False)
            I1_padded = pad_image_for_rife(current_frame_tensor, model_inference_scale_factor, use_fp16)

            interpolated_tensors = make_rife_inference(I0_padded, I1_padded, exp, model, model_inference_scale_factor) if exp > 0 else []

            for mid_tensor in interpolated_tensors:
                _mid_tensor_slice = mid_tensor[0, :, :output_h, :output_w]
                mid_np = (_mid_tensor_slice.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(str(frames_output_dir / f"frame_{saved_frame_count:07d}.png"), cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR))
                saved_frame_count += 1
            
            _current_I1_tensor_slice = I1_padded[0, :, :output_h, :output_w]
            current_I1_np_to_save = (_current_I1_tensor_slice.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            cv2.imwrite(str(frames_output_dir / f"frame_{saved_frame_count:07d}.png"), cv2.cvtColor(current_I1_np_to_save, cv2.COLOR_RGB2BGR))
            saved_frame_count += 1
            
            progress_bar.update(1)

        progress_bar.close()
        print(f"Frame generation complete. Total frames for output video: {saved_frame_count}")

        ffmpeg_cmd_create_video = [
            'ffmpeg', '-y', 
            '-r', str(final_fps), 
            '-i', str(frames_output_dir / 'frame_%07d.png'),
            '-s', f'{output_w}x{output_h}',
            '-c:v', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            '-movflags', '+faststart',
            final_output_video_path
        ]
        print("Creating video from interpolated frames...")
        
        success_video_creation, msg_video_creation = run_ffmpeg_command(ffmpeg_cmd_create_video, str(operation_dir))
        if not success_video_creation:
            raise RuntimeError(f"FFmpeg error during video creation: {msg_video_creation}")
        print(f"Video successfully created at {final_output_video_path}")

        print("Attempting audio transfer...")
        _, audio_msg = transfer_audio_ffmpeg(input_video_path, final_output_video_path, str(operation_dir))
        print(f"Audio transfer status: {audio_msg}")
        
        return final_output_video_path

    except Exception as e:
        if operation_dir and operation_dir.exists():
            shutil.rmtree(operation_dir)
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Video interpolation failed: {type(e).__name__} - {str(e)}")
    finally:
        if operation_dir and operation_dir.exists():
             shutil.rmtree(operation_dir)
        torch.set_default_dtype(original_torch_dtype) 