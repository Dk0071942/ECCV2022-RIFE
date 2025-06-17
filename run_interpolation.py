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

# Add the submodule's parent directory to sys.path to allow imports from the main project
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

MODEL_DIR = script_dir / 'train_log'

# Global model variable for RIFE
rife_loaded_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def cv2_frame_reader(video_path):
    """A generator to read video frames using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file at {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

def _run_ffmpeg_command(command, operation_dir_to_clean=None):
    try:
        # Add -loglevel error to suppress all but the most critical messages.
        # The command is copied to avoid modifying the original list.
        ffmpeg_command = command[:1] + ['-loglevel', 'error'] + command[1:]
        
        # The command print is removed for a cleaner log. For debugging, uncomment the line below.
        # print(f"FFmpeg CMD: {' '.join(ffmpeg_command)}")
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        # stdout/stderr printing on success is removed for cleaner logs.
        return True, "FFmpeg command successful."
    except subprocess.CalledProcessError as e:
        err_msg = f"FFmpeg Error (code {e.returncode}): {e.stderr.strip() if e.stderr else 'Unknown FFmpeg error.'}"
        print(err_msg)
        if operation_dir_to_clean and os.path.exists(operation_dir_to_clean):
            shutil.rmtree(operation_dir_to_clean)
        return False, err_msg[:1000]
    except FileNotFoundError:
        err_msg = "FFmpeg command not found. Ensure FFmpeg is installed and in system PATH."
        if operation_dir_to_clean and os.path.exists(operation_dir_to_clean):
            shutil.rmtree(operation_dir_to_clean)
        return False, err_msg
    except Exception as e:
        err_msg = f"An unexpected error occurred with FFmpeg: {str(e)}"
        if operation_dir_to_clean and os.path.exists(operation_dir_to_clean):
            shutil.rmtree(operation_dir_to_clean)
        return False, err_msg

def _transfer_audio_ffmpeg(source_video_path: str, target_video_path: str, operation_dir_for_cleanup: str = None):
    # This function remains largely the same as in your original implementation.
    op_dir = Path(operation_dir_for_cleanup or Path.cwd())
    op_dir.mkdir(exist_ok=True)

    temp_audio_file = op_dir / "temp_audio_for_transfer.mkv"
    target_video_no_audio = op_dir / "target_no_audio.mp4"

    cmd_extract_audio = ['ffmpeg', '-y', '-i', str(source_video_path), '-c:a', 'copy', '-vn', str(temp_audio_file)]
    success_extract, msg_extract = _run_ffmpeg_command(cmd_extract_audio, str(op_dir))
    if not success_extract:
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return False, f"Audio extraction failed: {msg_extract}. Output video will have no audio."

    try:
        target_video_path_obj = Path(target_video_path)
        if target_video_path_obj.exists():
            target_video_path_obj.rename(target_video_no_audio)
        else:
            if temp_audio_file.exists(): temp_audio_file.unlink()
            return False, "Target video for audio merge not found."
    except Exception as e:
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return False, f"Failed to rename target video for audio merge: {str(e)}"

    cmd_merge_audio = ['ffmpeg', '-y', '-i', str(target_video_no_audio), '-i', str(temp_audio_file), '-c', 'copy', str(target_video_path)]
    success_merge, msg_merge = _run_ffmpeg_command(cmd_merge_audio, str(op_dir))

    target_video_path_obj = Path(target_video_path)
    if not success_merge or (target_video_path_obj.exists() and target_video_path_obj.stat().st_size == 0):
        print(f"Lossless audio transfer failed ({msg_merge}). Retrying with AAC transcode...")
        temp_audio_file_aac = op_dir / "temp_audio_for_transfer.m4a"
        cmd_transcode_audio = ['ffmpeg', '-y', '-i', str(source_video_path), '-c:a', 'aac', '-b:a', '160k', '-vn', str(temp_audio_file_aac)]
        success_transcode, msg_transcode = _run_ffmpeg_command(cmd_transcode_audio, str(op_dir))

        if success_transcode:
            cmd_merge_aac = ['ffmpeg', '-y', '-i', str(target_video_no_audio), '-i', str(temp_audio_file_aac), '-c', 'copy', str(target_video_path)]
            success_merge_aac, msg_merge_aac = _run_ffmpeg_command(cmd_merge_aac, str(op_dir))
            if temp_audio_file_aac.exists(): temp_audio_file_aac.unlink()
            
            if success_merge_aac and target_video_path_obj.exists() and target_video_path_obj.stat().st_size > 0:
                if target_video_no_audio.exists(): target_video_no_audio.unlink()
                if temp_audio_file.exists(): temp_audio_file.unlink()
                return True, "Audio transferred with AAC transcode."
            else:
                target_video_no_audio.rename(target_video_path)
                if temp_audio_file.exists(): temp_audio_file.unlink()
                return False, f"AAC audio merge also failed ({msg_merge_aac})."
        else:
            target_video_no_audio.rename(target_video_path)
            if temp_audio_file.exists(): temp_audio_file.unlink()
            return False, f"Audio transcode to AAC failed ({msg_transcode})."
    else:
        if target_video_no_audio.exists(): target_video_no_audio.unlink()
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return True, "Audio transferred successfully (lossless)."

def get_rife_model():
    global rife_loaded_model
    if rife_loaded_model is not None:
        return rife_loaded_model
    
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        MODEL_DIR.mkdir(exist_ok=True)
        readme_path = MODEL_DIR / "README.md"
        error_message = (
            f"RIFE pre-trained models not found in '{MODEL_DIR}'.\n"
            f"Please download the models manually and place them in that directory.\n"
            f"A README with instructions has been created at: '{readme_path}'"
        )
        if not readme_path.exists():
            readme_content = f"""# RIFE Pre-trained Models
            
Download the models from [here](https://github.com/hzwer/ECCV2022-RIFE/releases/download/4.7/RIFE_trained_models_v4.7.zip)
or [here](https://drive.google.com/drive/folders/1i3_x4_PEp6F9L2D9Jc4_9aA8z-452j_H).

Extract the zip and place all its contents (folders like `RIFE_HDv2` and files like `flownet.pkl`) directly into this directory:
`{MODEL_DIR.resolve()}`
"""
            readme_path.write_text(readme_content)
        raise FileNotFoundError(error_message)

    try:
        from model.RIFE_HDv2 import Model
        m = Model()
        m.load_model(str(MODEL_DIR), -1)
        print("Loaded RIFE v2.x HD model.")
    except Exception:
        try:
            from train_log.RIFE_HDv3 import Model
            m = Model()
            m.load_model(str(MODEL_DIR), -1)
            print("Loaded RIFE v3.x HD model.")
        except Exception:
            try:
                from model.RIFE_HD import Model
                m = Model()
                m.load_model(str(MODEL_DIR), -1)
                print("Loaded RIFE v1.x HD model.")
            except Exception:
                from model.RIFE import Model
                m = Model()
                m.load_model(str(MODEL_DIR), -1)
                print("Loaded ArXiv-RIFE model.")
            
    m.eval()
    m.device()
    rife_loaded_model = m
    return rife_loaded_model

def _pad_image_for_video_interpolation(img_tensor, model_scale_factor, fp16_active):
    _n, _c, h_orig, w_orig = img_tensor.shape
    tmp = max(32, int(32 / model_scale_factor)) 
    ph = ((h_orig - 1) // tmp + 1) * tmp
    pw = ((w_orig - 1) // tmp + 1) * tmp
    padding = (0, pw - w_orig, 0, ph - h_orig)
    
    padded_tensor = F.pad(img_tensor, padding)
    if fp16_active:
        return padded_tensor.half()
    return padded_tensor

def _make_inference_for_video_interpolation(I0, I1, n_exp, model_instance, model_scale_factor_for_inference):
    if n_exp == 0:
        return []
    
    middle = model_instance.inference(I0, I1, scale=model_scale_factor_for_inference)
    
    if n_exp == 1:
        return [middle]

    first_half = _make_inference_for_video_interpolation(I0, middle, n_exp - 1, model_instance, model_scale_factor_for_inference)
    second_half = _make_inference_for_video_interpolation(middle, I1, n_exp - 1, model_instance, model_scale_factor_for_inference)
    
    return [*first_half, middle, *second_half]

def main_interpolate(
    input_video_path: str,
    output_dir_path: str, # Changed to accept a directory
    exp: int = 1,
    use_fp16: bool = False,
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

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError("Could not open input video.")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_input_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total_frames_input_video == 0:
            raise ValueError("Input video contains zero frames.")

        output_w, output_h = input_w, input_h
        model_inference_scale_factor = 1.0
        final_fps = original_fps * (2**exp)

        print(f"RIFE Input: {total_frames_input_video} frames, {original_fps:.2f} FPS, {input_w}x{input_h}")
        print(f"RIFE Output: Target FPS: {final_fps:.2f}, Resolution: {output_w}x{output_h}")
        
        videogen = cv2_frame_reader(input_video_path)
        last_frame_np = next(videogen).copy()

        last_frame_tensor = torch.from_numpy(last_frame_np.transpose(2, 0, 1)).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1_padded = _pad_image_for_video_interpolation(last_frame_tensor, model_inference_scale_factor, use_fp16)
        
        _first_frame_tensor_slice = I1_padded[0, :, :output_h, :output_w]
        first_frame_to_save_np = (_first_frame_tensor_slice.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite(str(frames_output_dir / f"frame_{0:07d}.png"), cv2.cvtColor(first_frame_to_save_np, cv2.COLOR_RGB2BGR))
        saved_frame_count = 1
        
        progress_bar = tqdm(total=total_frames_input_video - 1, desc="RIFE Interpolating Frames")

        for _, current_frame_np_orig in enumerate(videogen, start=1):
            current_frame_np = current_frame_np_orig.copy()
            I0_padded = I1_padded

            current_frame_tensor = torch.from_numpy(current_frame_np.transpose(2, 0, 1)).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1_padded = _pad_image_for_video_interpolation(current_frame_tensor, model_inference_scale_factor, use_fp16)

            interpolated_tensors = _make_inference_for_video_interpolation(I0_padded, I1_padded, exp, model, model_inference_scale_factor) if exp > 0 else []

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
        
        success_video_creation, msg_video_creation = _run_ffmpeg_command(ffmpeg_cmd_create_video, str(operation_dir))
        if not success_video_creation:
            raise RuntimeError(f"FFmpeg error during video creation: {msg_video_creation}")
        print(f"Video successfully created at {final_output_video_path}")

        print("Attempting audio transfer...")
        _, audio_msg = _transfer_audio_ffmpeg(input_video_path, final_output_video_path, str(operation_dir))
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