import gradio as gr
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

# --- Global Configuration & Setup ---
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

MODEL_DIR = script_dir / 'train_log'
TEMP_OUTPUT_DIR = script_dir / "temp_gradio_videos"
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# --- FFmpeg Frame Reader ---
def ffmpeg_frame_reader(video_path, width, height):
    """A generator to read video frames using a direct FFmpeg pipe."""
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', video_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-'  # Pipe to stdout
    ]
    # Use a buffer size that is a multiple of the frame size for efficiency
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=width*height*3*4)
    
    frame_size = width * height * 3
    
    try:
        while True:
            raw_frame = pipe.stdout.read(frame_size)
            if not raw_frame:
                break
            # The last frame might be incomplete
            if len(raw_frame) != frame_size:
                continue
                
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            yield frame
    finally:
        pipe.terminate()  # Ensure the ffmpeg process is killed
        pipe.wait()

# --- Model Loading ---
rife_loaded_model = None

def get_rife_model():
    """Loads the RIFE model, checking for model files and providing instructions if missing."""
    global rife_loaded_model
    if rife_loaded_model is not None:
        return rife_loaded_model

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        MODEL_DIR.mkdir(exist_ok=True)
        readme_path = MODEL_DIR / "README.md"
        error_message = (
            f"RIFE pre-trained models not found in '{MODEL_DIR}'.\n"
            f"Please download the models and place them in that directory.\n"
            f"A README with instructions has been created at: '{readme_path}'"
        )
        if not readme_path.exists():
            readme_content = f"""# RIFE Pre-trained Models
Download the models from the official RIFE repository. A common source is the v4.7 release:
[https://github.com/hzwer/ECCV2022-RIFE/releases/download/4.7/RIFE_trained_models_v4.7.zip](https://github.com/hzwer/ECCV2022-RIFE/releases/download/4.7/RIFE_trained_models_v4.7.zip)

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
                try:
                    from model.RIFE import Model
                    m = Model()
                    m.load_model(str(MODEL_DIR), -1)
                    print("Loaded ArXiv-RIFE model.")
                except Exception as e:
                    raise RuntimeError(f"Failed to load any RIFE model. Last error: {e}")

    m.eval()
    m.device()
    rife_loaded_model = m
    return rife_loaded_model

# --- Core Interpolation Logic (adapted from run_interpolation.py) ---

def _run_ffmpeg_command(command, operation_dir_to_clean=None):
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
        return False, err_msg
    except FileNotFoundError:
        err_msg = "FFmpeg command not found. Ensure FFmpeg is installed and in your system PATH."
        if operation_dir_to_clean and os.path.exists(operation_dir_to_clean):
            shutil.rmtree(operation_dir_to_clean)
        return False, err_msg
    return False, "An unexpected error occurred with FFmpeg."


def _transfer_audio_ffmpeg(source_video_path, target_video_path, op_dir_path):
    """Transfers audio from source to target video, with fallback to AAC."""
    op_dir = Path(op_dir_path)
    temp_audio_file = op_dir / "temp_audio.mkv"
    target_no_audio = op_dir / "target_no_audio.mp4"

    cmd_extract = ['ffmpeg', '-y', '-i', str(source_video_path), '-c:a', 'copy', '-vn', str(temp_audio_file)]
    if not _run_ffmpeg_command(cmd_extract)[0]:
        return False, "Audio extraction failed. Video will have no audio."

    if not Path(target_video_path).exists():
        return False, "Target video for audio merge not found."

    Path(target_video_path).rename(target_no_audio)

    cmd_merge = ['ffmpeg', '-y', '-i', str(target_no_audio), '-i', str(temp_audio_file), '-c', 'copy', str(target_video_path)]
    success, msg = _run_ffmpeg_command(cmd_merge)
    
    if success and Path(target_video_path).stat().st_size > 0:
        return True, "Audio transferred successfully (lossless)."

    print("Lossless audio transfer failed, retrying with AAC transcode...")
    temp_audio_aac = op_dir / "temp_audio.m4a"
    cmd_transcode = ['ffmpeg', '-y', '-i', str(source_video_path), '-c:a', 'aac', '-b:a', '160k', '-vn', str(temp_audio_aac)]
    if not _run_ffmpeg_command(cmd_transcode)[0]:
        target_no_audio.rename(target_video_path)
        return False, "Audio transcode to AAC failed."

    cmd_merge_aac = ['ffmpeg', '-y', '-i', str(target_no_audio), '-i', str(temp_audio_aac), '-c', 'copy', str(target_video_path)]
    if _run_ffmpeg_command(cmd_merge_aac)[0]:
        return True, "Audio transferred with AAC transcode."

    target_no_audio.rename(target_video_path)
    return False, "AAC audio merge also failed."

def _pad_image(img_tensor, scale_factor=1.0):
    """Pads image tensor to be divisible by 64 for RIFE model."""
    _n, _c, h, w = img_tensor.shape
    tmp = max(64, int(64 / scale_factor))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    return F.pad(img_tensor, padding)

def _make_inference(I0, I1, n_exp, model, scale):
    """Recursive inference function."""
    if n_exp == 0:
        return []
    middle = model.inference(I0, I1, scale)
    if n_exp == 1:
        return [middle]
    first_half = _make_inference(I0, middle, n_exp - 1, model, scale)
    second_half = _make_inference(middle, I1, n_exp - 1, model, scale)
    return [*first_half, middle, *second_half]

def interpolate_video(
    input_video_path,
    exp_factor,
    use_fp16,
    progress=gr.Progress(track_tqdm=True)
):
    """Main function to perform video interpolation, called by the Gradio interface."""
    if not input_video_path:
        raise gr.Error("Input video not provided.")
    
    try:
        model = get_rife_model()
    except (FileNotFoundError, RuntimeError) as e:
        raise gr.Error(str(e))

    if use_fp16 and not torch.cuda.is_available():
        raise gr.Error("FP16 is selected, but CUDA is not available.")

    operation_dir = None
    try:
        if use_fp16:
            torch.set_default_dtype(torch.float16)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        op_dir_name = f"interp_op_{Path(input_video_path).stem}_{timestamp}"
        operation_dir = TEMP_OUTPUT_DIR / op_dir_name
        os.makedirs(operation_dir, exist_ok=True)
        
        frames_output_dir = operation_dir / "frames"
        os.makedirs(frames_output_dir, exist_ok=True)

        final_output_path = TEMP_OUTPUT_DIR / f"{Path(input_video_path).stem}_interpolated_{exp_factor}x.mp4"

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened(): raise gr.Error("Could not open input video.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if frame_count == 0: raise gr.Error("Input video has zero frames.")

        try:
            videogen = ffmpeg_frame_reader(input_video_path, w, h)
            last_frame_np = next(videogen).copy()
        except StopIteration:
            raise gr.Error("Could not read any frames using FFmpeg. The video may be empty or corrupt.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Failed to start FFmpeg frame reader. Error: {str(e)}")

        last_frame_tensor = torch.from_numpy(last_frame_np.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
        I1_padded = _pad_image(last_frame_tensor)
        
        first_frame_np = (I1_padded[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[:h, :w]
        cv2.imwrite(str(frames_output_dir / "f_0000001.png"), cv2.cvtColor(first_frame_np, cv2.COLOR_RGB2BGR))
        
        frame_write_count = 1
        pbar = tqdm(total=frame_count-1, desc="Interpolating Frames")
        
        for current_frame_np_orig in videogen:
            current_frame_np = current_frame_np_orig.copy()
            I0_padded = I1_padded
            current_tensor = torch.from_numpy(current_frame_np.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
            I1_padded = _pad_image(current_tensor)
            
            if exp_factor > 0:
                interpolated_tensors = _make_inference(I0_padded, I1_padded, exp_factor, model, scale=1.0)
                for mid_tensor in interpolated_tensors:
                    frame_write_count += 1
                    mid_np = (mid_tensor[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[:h, :w]
                    cv2.imwrite(str(frames_output_dir / f"f_{frame_write_count:07d}.png"), cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR))
            
            frame_write_count += 1
            current_np = (I1_padded[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[:h, :w]
            cv2.imwrite(str(frames_output_dir / f"f_{frame_write_count:07d}.png"), cv2.cvtColor(current_np, cv2.COLOR_RGB2BGR))
            
            pbar.update(1)
        
        pbar.close()
        
        final_fps = fps * (2**exp_factor)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-r', str(final_fps),
            '-i', str(frames_output_dir / 'f_%07d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(final_output_path)
        ]
        
        success, msg = _run_ffmpeg_command(ffmpeg_cmd, operation_dir)
        if not success:
            raise gr.Error(f"FFmpeg video creation failed: {msg}")

        audio_success, audio_msg = _transfer_audio_ffmpeg(input_video_path, str(final_output_path), operation_dir)
        print(f"Audio transfer status: {audio_msg}")

        return str(final_output_path), f"Interpolation successful. {audio_msg}"

    except Exception as e:
        if operation_dir and os.path.exists(operation_dir):
            shutil.rmtree(operation_dir)
        if isinstance(e, gr.Error):
            raise e
        import traceback
        traceback.print_exc()
        raise gr.Error(f"An unexpected error occurred: {str(e)}")
    finally:
        if use_fp16:
            torch.set_default_dtype(torch.float32)
        if operation_dir and os.path.exists(operation_dir):
            shutil.rmtree(operation_dir)

# --- Gradio UI ---
def create_gradio_ui():
    """Defines and returns the Gradio UI blocks."""
    with gr.Blocks(title="RIFE Video Interpolation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Simple RIFE Video Frame Interpolation")
        gr.Markdown("Upload a video, choose the interpolation factor, and generate a slow-motion version.")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                exp_slider = gr.Slider(
                    minimum=1, maximum=4, value=1, step=1,
                    label="Interpolation Factor (2^x)",
                    info="1=2x frames, 2=4x, 3=8x, 4=16x."
                )
                fp16_checkbox = gr.Checkbox(
                    label="Use FP16 Inference (for NVIDIA GPUs)",
                    value=False,
                    info="Faster processing, less VRAM. Requires compatible GPU."
                )
                interpolate_button = gr.Button("Generate Interpolated Video", variant="primary")
            
            with gr.Column(scale=1):
                video_output = gr.Video(label="Interpolated Video")
                status_textbox = gr.Textbox(label="Status", interactive=False)

        interpolate_button.click(
            fn=interpolate_video,
            inputs=[video_input, exp_slider, fp16_checkbox],
            outputs=[video_output, status_textbox]
        )
        
        gr.Markdown("---")
        gr.Markdown(
            """**Notes:**
- **FFmpeg:** Must be installed and in your system's PATH.
- **Models:** RIFE models must be downloaded into the `train_log` directory. If not found, a `README.md` file with instructions will be created there.
- **Temporary Files:** Generated videos are saved in the `temp_gradio_videos` folder and temporary processing files are deleted after each run."""
        )
    return demo

if __name__ == '__main__':
    # Pre-flight checks
    if shutil.which("ffmpeg") is None:
        print("\n" + "="*80)
        print("WARNING: ffmpeg was not found in your system's PATH.")
        print("The Gradio app will load, but video processing will fail.")
        print("Please install FFmpeg and ensure it's available in your system's PATH.")
        print("="*80 + "\n")

    try:
        get_rife_model()
        print("RIFE model loaded successfully.")
    except (FileNotFoundError, RuntimeError) as e:
        print("\n" + "="*80)
        print(f"MODEL LOADING ERROR: {e}")
        print("The Gradio app will run, but interpolation will fail until models are correctly placed.")
        print("="*80 + "\n")

    app = create_gradio_ui()
    app.launch() 