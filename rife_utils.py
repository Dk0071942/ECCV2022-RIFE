import os
import cv2
import torch
import numpy as np
from torch.nn import functional as F
import shutil
import subprocess
from pathlib import Path
import sys

# Add the submodule's parent directory to sys.path to allow imports from the main project
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# Global model variable for RIFE
rife_loaded_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# --- Video/System Helpers ---

def cv2_frame_reader(video_path):
    """A generator to read video frames using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file at {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

def get_video_properties(video_path):
    """Gets video properties using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file at {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count

def run_ffmpeg_command(command, operation_dir_to_clean=None):
    """Executes an FFmpeg command, handling errors and cleanup."""
    try:
        ffmpeg_command = command[:1] + ['-loglevel', 'error'] + command[1:]
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
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

def transfer_audio_ffmpeg(source_video_path: str, target_video_path: str, operation_dir_for_cleanup: str = None):
    """Transfers audio from a source video to a target video using FFmpeg."""
    op_dir = Path(operation_dir_for_cleanup or Path.cwd())
    op_dir.mkdir(exist_ok=True)

    temp_audio_file = op_dir / "temp_audio_for_transfer.mkv"
    target_video_no_audio = op_dir / "target_no_audio.mp4"

    cmd_extract_audio = ['ffmpeg', '-y', '-i', str(source_video_path), '-c:a', 'copy', '-vn', str(temp_audio_file)]
    success_extract, msg_extract = run_ffmpeg_command(cmd_extract_audio, str(op_dir))
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
    success_merge, msg_merge = run_ffmpeg_command(cmd_merge_audio, str(op_dir))

    target_video_path_obj = Path(target_video_path)
    if not success_merge or (target_video_path_obj.exists() and target_video_path_obj.stat().st_size == 0):
        temp_audio_file_aac = op_dir / "temp_audio_for_transfer.m4a"
        cmd_transcode_audio = ['ffmpeg', '-y', '-i', str(source_video_path), '-c:a', 'aac', '-b:a', '160k', '-vn', str(temp_audio_file_aac)]
        success_transcode, _ = run_ffmpeg_command(cmd_transcode_audio, str(op_dir))

        if success_transcode:
            cmd_merge_aac = ['ffmpeg', '-y', '-i', str(target_video_no_audio), '-i', str(temp_audio_file_aac), '-c', 'copy', str(target_video_path)]
            success_merge_aac, msg_merge_aac = run_ffmpeg_command(cmd_merge_aac, str(op_dir))
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
            return False, "Audio transcode to AAC failed."
    else:
        if target_video_no_audio.exists(): target_video_no_audio.unlink()
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return True, "Audio transferred successfully (lossless)."

# --- Model Loading and Inference Helpers ---

def get_rife_model():
    """Loads the RIFE model, checking for pre-trained files."""
    global rife_loaded_model
    if rife_loaded_model is not None:
        return rife_loaded_model
    
    model_dir = script_dir / 'train_log'
    if not model_dir.exists() or not any(model_dir.iterdir()):
        model_dir.mkdir(exist_ok=True)
        readme_path = model_dir / "README.md"
        error_message = (
            f"RIFE pre-trained models not found in '{model_dir}'.\n"
            f"Please download the models manually and place them in that directory.\n"
            f"A README with instructions has been created at: '{readme_path}'"
        )
        if not readme_path.exists():
            readme_content = f"""# RIFE Pre-trained Models
            
Download the models from [here](https://github.com/hzwer/ECCV2022-RIFE/releases/download/4.7/RIFE_trained_models_v4.7.zip)
or [here](https://drive.google.com/drive/folders/1i3_x4_PEp6F9L2D9Jc4_9aA8z-452j_H).

Extract the zip and place all its contents (folders like `RIFE_HDv2` and files like `flownet.pkl`) directly into this directory:
`{model_dir.resolve()}`
"""
            readme_path.write_text(readme_content)
        raise FileNotFoundError(error_message)

    try:
        from model.RIFE_HDv2 import Model
        m = Model()
        m.load_model(str(model_dir), -1)
        print("Loaded RIFE v2.x HD model.")
    except Exception:
        try:
            from train_log.RIFE_HDv3 import Model
            m = Model()
            m.load_model(str(model_dir), -1)
            print("Loaded RIFE v3.x HD model.")
        except Exception:
            try:
                from model.RIFE_HD import Model
                m = Model()
                m.load_model(str(model_dir), -1)
                print("Loaded RIFE v1.x HD model.")
            except Exception:
                from model.RIFE import Model
                m = Model()
                m.load_model(str(model_dir), -1)
                print("Loaded ArXiv-RIFE model.")
            
    m.eval()
    m.device()
    rife_loaded_model = m
    return rife_loaded_model

def pad_image_for_rife(img_tensor, model_scale_factor, fp16_active):
    """Pads an image tensor to be compatible with RIFE model inference dimensions."""
    _n, _c, h_orig, w_orig = img_tensor.shape
    tmp = max(64, int(64 / model_scale_factor)) 
    ph = ((h_orig - 1) // tmp + 1) * tmp
    pw = ((w_orig - 1) // tmp + 1) * tmp
    padding = (0, pw - w_orig, 0, ph - h_orig)
    
    padded_tensor = F.pad(img_tensor, padding)
    if fp16_active:
        return padded_tensor.half()
    return padded_tensor

def make_rife_inference(I0, I1, n_exp, model_instance, model_scale_factor_for_inference):
    """Recursively generates n intermediate frames between I0 and I1."""
    if n_exp == 0:
        return []
    
    middle = model_instance.inference(I0, I1, scale=model_scale_factor_for_inference)
    
    if n_exp == 1:
        return [middle]

    first_half = make_rife_inference(I0, middle, n_exp - 1, model_instance, model_scale_factor_for_inference)
    second_half = make_rife_inference(middle, I1, n_exp - 1, model_instance, model_scale_factor_for_inference)
    
    return [*first_half, middle, *second_half] 