import gradio as gr
import os
import cv2
import torch
import numpy as np
from PIL import Image # Added for PIL Image conversion
from torch.nn import functional as F
import shutil
import subprocess
import datetime
# import time # time was imported but not used directly, datetime covers timestamps

# --- Configuration & Model Loading ---
MODEL_DIR = 'train_log'
TEMP_IMAGE_DIR_PARENT = "temp_gradio_frames" # Parent for temporary frame storage for interpolation
TEMP_VIDEO_DIR = "temp_gradio_videos"       # Temporary video storage for interpolation output
TEMP_CHAINED_OP_DIR = "temp_chained_operations"

# Ensure base temporary directories exist
os.makedirs(TEMP_IMAGE_DIR_PARENT, exist_ok=True)
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
os.makedirs(TEMP_CHAINED_OP_DIR, exist_ok=True)
# os.makedirs(TEMP_EXTRACTED_FRAMES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# Global model variable
loaded_model = None

def get_model():
    global loaded_model
    if loaded_model is not None:
        return loaded_model

    # Try to load the model (same logic as inference_img.py and inference_video.py)
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                m = Model()
                m.load_model(MODEL_DIR, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model # Common case from user logs
                m = Model()
                m.load_model(MODEL_DIR, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            m = Model()
            m.load_model(MODEL_DIR, -1)
            print("Loaded v1.x HD model")
        loaded_model = m
    except Exception as e:
        try:
            from model.RIFE import Model # Fallback to arxiv RIFE
            m = Model()
            m.load_model(MODEL_DIR, -1)
            print("Loaded ArXiv-RIFE model")
            loaded_model = m
        except Exception as e_arxiv:
            error_message = f"Error loading any RIFE model. Main error: {e}, Arxiv fallback error: {e_arxiv}. Check MODEL_DIR ('{MODEL_DIR}') and ensure model files (e.g., RIFE_HDv3.py) are in the correct location (e.g., 'train_log' or 'model' directory) and importable."
            print(error_message)
            raise RuntimeError(error_message) # Stop app launch if model fails

    loaded_model.eval()
    loaded_model.device()
    return loaded_model

# --- Internal Helper for Interpolated Frame Generation & Saving ---
def _generate_and_save_interpolated_frames(img0_pil, img1_pil, exp_value, target_frame_directory, model_instance):
    os.makedirs(target_frame_directory, exist_ok=True)
    try:
        img0_cv_rgb = np.array(img0_pil)
        img1_cv_rgb = np.array(img1_pil)
        img0_cv_bgr = cv2.cvtColor(img0_cv_rgb, cv2.COLOR_RGB2BGR)
        img1_cv_bgr = cv2.cvtColor(img1_cv_rgb, cv2.COLOR_RGB2BGR)

        img0_tensor_chw = torch.from_numpy(img0_cv_bgr.transpose(2, 0, 1)).float().to(device) / 255.
        img1_tensor_chw = torch.from_numpy(img1_cv_bgr.transpose(2, 0, 1)).float().to(device) / 255.
        img0_tensor_4d = img0_tensor_chw.unsqueeze(0)
        img1_tensor_4d = img1_tensor_chw.unsqueeze(0)

        _n, _c, h_orig, w_orig = img0_tensor_4d.shape # These are from the input PIL images, which might have been scaled already
        
        # Padding for RIFE model (multiples of 32)
        ph_model = ((h_orig - 1) // 32 + 1) * 32
        pw_model = ((w_orig - 1) // 32 + 1) * 32
        model_padding = (0, pw_model - w_orig, 0, ph_model - h_orig)
        
        img0_padded_model = F.pad(img0_tensor_4d, model_padding)
        img1_padded_model = F.pad(img1_tensor_4d, model_padding)

    except Exception as e:
        return False, f"Preprocessing for RIFE error: {str(e)}", None, (0,0)

    interpolated_frame_tensors = [img0_padded_model, img1_padded_model]
    for _ in range(exp_value):
        current_pass_frames_tensors = []
        for j in range(len(interpolated_frame_tensors) - 1):
            current_pass_frames_tensors.append(interpolated_frame_tensors[j])
            try:
                with torch.no_grad():
                    mid_frame = model_instance.inference(interpolated_frame_tensors[j], interpolated_frame_tensors[j + 1])
            except Exception as e:
                return False, f"RIFE inference error: {str(e)}", None, (w_orig,h_orig)
            current_pass_frames_tensors.append(mid_frame)
        current_pass_frames_tensors.append(img1_padded_model)
        interpolated_frame_tensors = current_pass_frames_tensors

    try:
        saved_frame_paths = []
        for i_frame, frame_tensor_4d in enumerate(interpolated_frame_tensors):
            frame_filename = os.path.join(target_frame_directory, f'frame_{i_frame:05d}.png')
            # Tensor is [1, C, H_padded_model, W_padded_model]. Crop to original H,W before saving.
            img_to_save_permuted = frame_tensor_4d[0].cpu().numpy().transpose(1, 2, 0)
            img_to_save_cropped = img_to_save_permuted[:h_orig, :w_orig, :] 
            img_to_save_uint8 = (img_to_save_cropped * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(frame_filename, img_to_save_uint8) # Save BGR data directly
            saved_frame_paths.append(frame_filename)
        return True, "Frames generated successfully.", saved_frame_paths, (w_orig, h_orig)
    except Exception as e:
        return False, f"Frame saving error: {str(e)}", None, (w_orig,h_orig)

# --- Video Info & Frame Extraction (for UI Tab 1) ---
def get_video_info(video_file_path):
    if video_file_path is None:
        info_text = "Video not loaded or cleared. Please upload a video."
        start_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        end_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        return info_text, start_update, end_update

    try:
        video_path = video_file_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            info_text = f"Error: Could not open video file: {video_path}"
            start_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
            end_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
            return info_text, start_update, end_update
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if total_frames > 0:
            info_text = f"Video Info: {total_frames} frames, {fps:.2f} FPS. Ready to extract."
            start_update = gr.update(minimum=1, maximum=total_frames, value=1, interactive=True)
            end_val = total_frames # Default end frame to total_frames
            if total_frames == 1: # If only one frame, end value must also be 1.
                 end_val = 1      # Extraction logic will prevent start >= end for actual extraction.
            end_update = gr.update(minimum=1, maximum=total_frames, value=end_val, interactive=True)
        else: # total_frames is 0 or video couldn't be properly read for frames
            info_text = f"Video Info: Could not determine frame count (0 frames found). FPS: {fps:.2f}. Cannot extract."
            start_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
            end_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        
        return info_text, start_update, end_update
    except Exception as e:
        info_text = f"Error reading video info: {str(e)}"
        start_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        end_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        return info_text, start_update, end_update

def extract_frames_from_video(video_file_path, start_frame_one_indexed_str, end_frame_one_indexed_str):
    if video_file_path is None:
        raise gr.Error("Please upload a video first.")

    try:
        start_frame_one_indexed = int(start_frame_one_indexed_str)
        end_frame_one_indexed = int(end_frame_one_indexed_str)
    except ValueError:
        raise gr.Error("Frame numbers must be integers.")

    video_path = video_file_path
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Error: Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not (1 <= start_frame_one_indexed <= total_frames):
        cap.release()
        raise gr.Error(f"Start frame ({start_frame_one_indexed}) is out of bounds (1-{total_frames}).")
    if not (1 <= end_frame_one_indexed <= total_frames):
        cap.release()
        raise gr.Error(f"End frame ({end_frame_one_indexed}) is out of bounds (1-{total_frames}).")
    if start_frame_one_indexed >= end_frame_one_indexed:
        cap.release()
        # For a single frame video, this makes extraction impossible, which is intended.
        # If total_frames == 1, start and end will both be 1, triggering this.
        raise gr.Error(f"Start frame ({start_frame_one_indexed}) must be strictly less than end frame ({end_frame_one_indexed}). To process a single frame video, consider different tooling or logic.")

    start_frame_0_indexed = start_frame_one_indexed - 1
    end_frame_0_indexed = end_frame_one_indexed - 1
    
    pil_start_frame, pil_end_frame = None, None

    try:
        # Extract start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_0_indexed)
        ret_start, frame_start_bgr = cap.read()
        if not ret_start:
            raise gr.Error(f"Could not read start frame ({start_frame_one_indexed}).")
        frame_start_rgb = cv2.cvtColor(frame_start_bgr, cv2.COLOR_BGR2RGB)
        pil_start_frame = Image.fromarray(frame_start_rgb)

        # Extract end frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame_0_indexed)
        ret_end, frame_end_bgr = cap.read()
        if not ret_end:
            raise gr.Error(f"Could not read end frame ({end_frame_one_indexed}).")
        frame_end_rgb = cv2.cvtColor(frame_end_bgr, cv2.COLOR_BGR2RGB)
        pil_end_frame = Image.fromarray(frame_end_rgb)
        
    except Exception as e:
        raise gr.Error(f"Error during frame extraction: {str(e)}")
    finally:
        cap.release()

    return pil_start_frame, pil_end_frame, pil_start_frame, pil_end_frame, f"Successfully extracted frames {start_frame_one_indexed} and {end_frame_one_indexed}."

# --- Standard Interpolation (for UI Tab 2) ---
def create_standard_interpolated_video(img0_pil, img1_pil, exp_value, fps):
    if img0_pil is None or img1_pil is None: raise gr.Error("Upload both images.")
    model = get_model()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_op_dir = os.path.join(TEMP_IMAGE_DIR_PARENT, f"std_interp_{timestamp}")
    frames_dir = os.path.join(unique_op_dir, "frames")
    
    success, message, saved_frame_paths, (w,h) = _generate_and_save_interpolated_frames(
        img0_pil, img1_pil, exp_value, frames_dir, model
    )
    if not success:
        if os.path.exists(unique_op_dir): shutil.rmtree(unique_op_dir)
        raise gr.Error(message)

    output_video_path = os.path.join(TEMP_VIDEO_DIR, f"std_slomo_{timestamp}.mp4")
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-r', str(fps), '-i', os.path.join(frames_dir, 'frame_%05d.png'),
        '-s', f'{w}x{h}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output_video_path
    ]
    success, msg = _run_ffmpeg_command(ffmpeg_cmd, unique_op_dir)
    if not success:
        # unique_op_dir already cleaned by _run_ffmpeg_command on failure
        raise gr.Error(f"FFmpeg error (standard): {msg}")
    # Clean up on success too
    if os.path.exists(unique_op_dir): shutil.rmtree(unique_op_dir)
    return output_video_path, "Standard interpolation successful."

# --- Chained Interpolation (for UI Tab 3) ---
def create_chained_interpolated_video(anchor_img_pil, middle_video_path, exp_value, interp_duration_seconds, main_video_final_fps):
    if anchor_img_pil is None: raise gr.Error("Anchor Image required.")
    if middle_video_path is None: raise gr.Error("Middle Video required.")
    if interp_duration_seconds <= 0: raise gr.Error("Interpolation duration must be positive.")
    model = get_model()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    operation_dir = os.path.join(TEMP_CHAINED_OP_DIR, f"chained_{timestamp}") # Unique parent for this whole operation
    os.makedirs(operation_dir, exist_ok=True)

    # --- Subdirectories for this operation ---
    temp_img_files_dir = os.path.join(operation_dir, "temp_images") # For scaled/padded images
    frames_interp1_dir = os.path.join(operation_dir, "frames_interp1")
    frames_interp2_dir = os.path.join(operation_dir, "frames_interp2")
    video_segments_dir = os.path.join(operation_dir, "video_segments")
    for d in [temp_img_files_dir, frames_interp1_dir, frames_interp2_dir, video_segments_dir]:
        os.makedirs(d, exist_ok=True)

    try:
        # 1. Determine Target Resolution from Middle Video
        cap_middle_info = cv2.VideoCapture(middle_video_path)
        if not cap_middle_info.isOpened(): raise gr.Error("Cannot open middle video for info.")
        target_w = int(cap_middle_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_h = int(cap_middle_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        middle_total_frames = int(cap_middle_info.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_middle_info.release()
        if target_w == 0 or target_h == 0 or middle_total_frames == 0: raise gr.Error("Middle video has invalid dimensions or zero frames.")

        # 2. Prepare Anchor Image (Scale/Pad to Target Resolution)
        temp_anchor_orig_path = os.path.join(temp_img_files_dir, "anchor_orig.png")
        anchor_img_pil.save(temp_anchor_orig_path)
        scaled_anchor_path = os.path.join(temp_img_files_dir, "anchor_scaled.png")
        success, msg = _scale_and_pad_image_ffmpeg(temp_anchor_orig_path, target_w, target_h, scaled_anchor_path, operation_dir)
        if not success: raise gr.Error(f"Scaling anchor image failed: {msg}")
        scaled_anchor_pil = Image.open(scaled_anchor_path)

        # 3. Extract and Prepare Middle Video's First/Last Frames
        cap_extract = cv2.VideoCapture(middle_video_path)
        if not cap_extract.isOpened(): raise gr.Error("Cannot open middle video for frame extraction.")
        
        # First frame
        cap_extract.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, vid_first_bgr = cap_extract.read()
        if not ret: raise gr.Error("Failed to read first frame of middle video.")
        temp_vid_first_orig_path = os.path.join(temp_img_files_dir, "vid_first_orig.png")
        cv2.imwrite(temp_vid_first_orig_path, vid_first_bgr)
        scaled_vid_first_path = os.path.join(temp_img_files_dir, "vid_first_scaled.png")
        success, msg = _scale_and_pad_image_ffmpeg(temp_vid_first_orig_path, target_w, target_h, scaled_vid_first_path, operation_dir)
        if not success: raise gr.Error(f"Scaling middle video first frame: {msg}")
        scaled_vid_first_pil = Image.open(scaled_vid_first_path)

        # Last frame
        cap_extract.set(cv2.CAP_PROP_POS_FRAMES, middle_total_frames - 1 if middle_total_frames > 0 else 0)
        ret, vid_last_bgr = cap_extract.read()
        if not ret: raise gr.Error("Failed to read last frame of middle video.")
        cap_extract.release()
        temp_vid_last_orig_path = os.path.join(temp_img_files_dir, "vid_last_orig.png")
        cv2.imwrite(temp_vid_last_orig_path, vid_last_bgr)
        scaled_vid_last_path = os.path.join(temp_img_files_dir, "vid_last_scaled.png")
        success, msg = _scale_and_pad_image_ffmpeg(temp_vid_last_orig_path, target_w, target_h, scaled_vid_last_path, operation_dir)
        if not success: raise gr.Error(f"Scaling middle video last frame: {msg}")
        scaled_vid_last_pil = Image.open(scaled_vid_last_path)

        # 4. Calculate Interpolation Segment FPS
        num_frames_per_interp_segment = (2**exp_value) + 1
        interp_segment_fps = num_frames_per_interp_segment / interp_duration_seconds
        if interp_segment_fps < 1: print(f"Warning: Calculated interpolation FPS is {interp_segment_fps:.2f}, may result in slow motion.")

        # 5. Interpolation 1 (Scaled Anchor -> Scaled Vid First)
        success, msg, _, _ = _generate_and_save_interpolated_frames(
            scaled_anchor_pil, scaled_vid_first_pil, exp_value, frames_interp1_dir, model)
        if not success: raise gr.Error(f"Interp1 gen failed: {msg}")
        interp1_segment_path = os.path.join(video_segments_dir, "interp1.mp4")
        cmd1 = ['ffmpeg', '-y', '-r', str(interp_segment_fps), '-i', os.path.join(frames_interp1_dir, 'frame_%05d.png'), '-s', f'{target_w}x{target_h}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', interp1_segment_path]
        success, msg = _run_ffmpeg_command(cmd1, operation_dir)
        if not success: raise gr.Error(f"FFmpeg for Interp1 video: {msg}")

        # 6. Interpolation 2 (Scaled Vid Last -> Scaled Anchor)
        success, msg, _, _ = _generate_and_save_interpolated_frames(
            scaled_vid_last_pil, scaled_anchor_pil, exp_value, frames_interp2_dir, model)
        if not success: raise gr.Error(f"Interp2 gen failed: {msg}")
        interp2_segment_path = os.path.join(video_segments_dir, "interp2.mp4")
        cmd2 = ['ffmpeg', '-y', '-r', str(interp_segment_fps), '-i', os.path.join(frames_interp2_dir, 'frame_%05d.png'), '-s', f'{target_w}x{target_h}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', interp2_segment_path]
        success, msg = _run_ffmpeg_command(cmd2, operation_dir)
        if not success: raise gr.Error(f"FFmpeg for Interp2 video: {msg}")

        # 7. Prepare Middle Video Segment (Re-encode to target FPS and target resolution)
        middle_reencoded_path = os.path.join(video_segments_dir, "middle_reencoded.mp4")
        # Use scale and pad for the middle video as well to ensure it matches target_w, target_h correctly
        vf_middle_scale_pad = f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=1,pad=w={target_w}:h={target_h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black"
        cmd_middle = [
            'ffmpeg', '-y', '-i', middle_video_path, '-vf', vf_middle_scale_pad,
            '-r', str(main_video_final_fps), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', middle_reencoded_path
        ]
        success, msg = _run_ffmpeg_command(cmd_middle, operation_dir)
        if not success: raise gr.Error(f"FFmpeg for Middle video prep: {msg}")

        # 8. Concatenate segments
        concat_list_path = os.path.join(operation_dir, "concat_list.txt")
        # Ensure relative paths for concat list if ffmpeg working dir is operation_dir, or use absolute paths.
        # Using absolute paths is safer here.
        with open(concat_list_path, 'w') as f:
            f.write(f"file '{os.path.abspath(interp1_segment_path)}'\n")
            f.write(f"file '{os.path.abspath(middle_reencoded_path)}'\n")
            f.write(f"file '{os.path.abspath(interp2_segment_path)}'\n")

        final_video_path = os.path.join(TEMP_VIDEO_DIR, f"chained_output_{timestamp}.mp4")
        cmd_concat = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', str(main_video_final_fps), '-movflags', '+faststart', final_video_path
        ]
        success, msg = _run_ffmpeg_command(cmd_concat, operation_dir) # operation_dir cleaned on failure
        if not success: raise gr.Error(f"FFmpeg for Concatenation: {msg}")

        return final_video_path, "Chained interpolation successful."
    
    except gr.Error as e: # Catch Gradio errors to avoid cleaning and re-raise
        # If operation_dir was passed to a failing _run_ffmpeg_command, it might be cleaned already.
        # So, only clean if it still exists AND the error is from our explicit raises before ffmpeg calls.
        if os.path.exists(operation_dir): shutil.rmtree(operation_dir)
        raise e 
    except Exception as e: # Catch any other unexpected Python errors
        if os.path.exists(operation_dir): shutil.rmtree(operation_dir)
        #raise gr.Error(f"Unexpected Chained Interpolation Error: {str(e)}")
        # For better debugging, re-raise the original exception if it's not a gr.Error
        print(f"Unexpected Chained Interpolation Error: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Internal error during chained interpolation. Check console. ({type(e).__name__})")
    finally:
        # Final cleanup, in case an error occurred that didn't trigger specific cleanup paths
        if os.path.exists(operation_dir): 
             print(f"Final cleanup: Removing operation directory {operation_dir}")
             shutil.rmtree(operation_dir)

def _run_ffmpeg_command(command, operation_dir_to_clean=None):
    try:
        print(f"FFmpeg CMD: {' '.join(command)}")
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        if process.stdout: print(f"FFmpeg STDOUT: {process.stdout.strip()}")
        if process.stderr: print(f"FFmpeg STDERR: {process.stderr.strip()}") # Often has info, not just errors
        return True, "FFmpeg command successful."
    except subprocess.CalledProcessError as e:
        err_msg = f"FFmpeg Error (code {e.returncode}): {e.stderr.strip() if e.stderr else 'Unknown FFmpeg error.'}"
        print(err_msg) # Also print to console for easier debugging
        if operation_dir_to_clean and os.path.exists(operation_dir_to_clean):
            shutil.rmtree(operation_dir_to_clean)
        # raise gr.Error(err_msg[:1000]) # This will be handled by the calling function
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

def _scale_and_pad_image_ffmpeg(input_img_path, target_w, target_h, output_img_path, operation_dir_to_clean=None):
    # Scale to fit within target_w x target_h, preserve aspect ratio, then pad to target_w x target_h
    # vf_scale_pad = f"scale=w='min(iw*{target_h}/ih,{target_w})':h='min(ih*{target_w}/iw,{target_h})':force_original_aspect_ratio=decrease,pad=w={target_w}:h={target_h}:x='(ow-iw)/2':y='(oh-ih)/2':color=black"
    # Simpler scale to fit, then pad. This version first scales to make one dimension fit, then pads the other if needed after ensuring aspect ratio.
    vf_filter = f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=1,pad=w={target_w}:h={target_h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black"

    command = [
        'ffmpeg', '-y', '-i', input_img_path,
        '-vf', vf_filter,
        '-pix_fmt', 'rgb24', # Ensure output PNG is standard RGB
        output_img_path
    ]
    return _run_ffmpeg_command(command, operation_dir_to_clean)

# --- Gradio Interface Definition with Blocks ---
with gr.Blocks(title="RIFE Video and Image Interpolation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RIFE Video and Image Frame Interpolation")
    gr.Markdown(
        """Use this tool to either extract start/end frames from a video OR directly upload two images, 
        then interpolate between them to create a slow-motion video."""
    )

    with gr.Tabs():
        with gr.TabItem("1. Select Frames from Video (Optional)"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video")
                    video_info_display = gr.Textbox(label="Video Information", interactive=False)
                    start_frame_num_input = gr.Number(label="Start Frame (1-indexed)", value=1, minimum=1, maximum=1, precision=0, interactive=False)
                    end_frame_num_input = gr.Number(label="End Frame (1-indexed)", value=1, minimum=1, maximum=1, precision=0, interactive=False)
                    extract_button = gr.Button("Extract & Use Frames for Interpolation", variant="secondary")
                    extraction_status_display = gr.Textbox(label="Extraction Status", interactive=False, show_label=False)
                with gr.Column(scale=1):
                    gr.Markdown("### Extracted Frames:")
                    extracted_start_img_display = gr.Image(label="Start Frame Preview", type="pil", interactive=False)
                    extracted_end_img_display = gr.Image(label="End Frame Preview", type="pil", interactive=False)
        
        with gr.TabItem("2. Interpolate Between Images"):
            gr.Markdown(
                """Upload two images below, or use frames extracted from the 'Select Frames from Video' tab. 
                These images will be used for interpolation."""
            )
            with gr.Row():
                img0_input_interpolation = gr.Image(type="pil", label="First Image for Interpolation")
                img1_input_interpolation = gr.Image(type="pil", label="Second Image for Interpolation")
            
            exp_slider = gr.Slider(
                minimum=1, maximum=5, value=2, step=1, 
                label="Interpolation Factor (exp)", 
                info="Generates 2^exp segments (e.g., exp=2 -> 2^2=4 segments, meaning 3 interpolated frames between originals, total 5 frames)."
            )
            fps_number = gr.Number(value=24, label="Output Video FPS", minimum=1, step=1)
            interpolate_button = gr.Button("Generate Interpolated Video", variant="primary")
            
            gr.Markdown("### Interpolated Video Output:")
            video_output_display = gr.Video(label="Output Video")
            interpolation_status_display = gr.Textbox(label="Interpolation Status", interactive=False, show_label=False)

        with gr.TabItem("3. Chained Video Interpolation (Image-Video-Image)"):
            gr.Markdown("Create a sequence: Anchor Image -> Interpolation -> Middle Video -> Interpolation -> Anchor Image.")
            with gr.Row():
                anchor_img_chained = gr.Image(type="pil", label="Anchor Image (Start/End)")
                middle_video_chained = gr.Video(label="Middle Video")
            exp_chained_interp = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Interpolation Factor (exp for both sides)")
            interp_duration_chained = gr.Number(value=2.0, minimum=0.1, label="Interpolation Segment Duration (seconds)", info="Duration for 'Anchor to Video Start' and 'Video End to Anchor' parts.")
            main_video_fps_chained = gr.Number(value=24, label="Main Video & Final Output FPS", minimum=1, step=1)
            btn_run_chained_interp = gr.Button("Generate Chained Interpolated Video", variant="primary")
            vid_output_chained_interp = gr.Video(label="Chained Output Video")
            status_chained_interp = gr.Textbox(label="Status", interactive=False, show_label=False)

    # --- Event Handlers ---
    video_input.upload(
        fn=get_video_info,
        inputs=[video_input],
        outputs=[video_info_display, start_frame_num_input, end_frame_num_input]
    )
    video_input.clear(
        fn=get_video_info, # Call with None input to reset frame selectors
        inputs=[video_input], # video_input will be None here
        outputs=[video_info_display, start_frame_num_input, end_frame_num_input]
    )

    extract_button.click(
        fn=extract_frames_from_video,
        inputs=[video_input, start_frame_num_input, end_frame_num_input],
        outputs=[
            extracted_start_img_display, 
            extracted_end_img_display,
            img0_input_interpolation, # Populate the interpolation input
            img1_input_interpolation, # Populate the interpolation input
            extraction_status_display
        ]
    )

    interpolate_button.click(
        fn=create_standard_interpolated_video,
        inputs=[img0_input_interpolation, img1_input_interpolation, exp_slider, fps_number],
        outputs=[video_output_display, interpolation_status_display]
    )

    btn_run_chained_interp.click(
        fn=create_chained_interpolated_video,
        inputs=[anchor_img_chained, middle_video_chained, exp_chained_interp, interp_duration_chained, main_video_fps_chained],
        outputs=[vid_output_chained_interp, status_chained_interp]
    )
    
    gr.Markdown("---")
    gr.Markdown(
        """**Important Notes:**

- **FFmpeg Installation:** Ensure FFmpeg is installed and accessible in your system's PATH for video generation.

- **Model Files:** RIFE model files (e.g., `RIFE_HDv3.py` and `.pth` weights) must be in the `'train_log'` directory 
(or the correct subdirectory as per model imports) relative to where you run this script.

- **Processing Time:** Interpolation can be resource-intensive and may take time depending on image size and the interpolation factor."""
    )
    article = "<p style='text-align: center'><a href='https://github.com/hzwer/Practical-RIFE' target='_blank'>Practical-RIFE GitHub Repository</a></p>"
    gr.Markdown(article)


if __name__ == '__main__':
    # Pre-check for FFmpeg
    if shutil.which("ffmpeg") is None:
        print("---------------------------------------------------------------------------")
        print("WARNING: ffmpeg was not found in your system's PATH.")
        print("The Gradio app will load, but video generation will likely fail.")
        print("Please install FFmpeg and ensure it is added to your system's PATH.")
        print("---------------------------------------------------------------------------")
    
    # Attempt to load the model once at startup to catch errors early
    try:
        get_model()
        print("RIFE model loaded successfully for Gradio app.")
    except RuntimeError as e:
        print(f"Failed to load RIFE model at startup: {e}")
        print("The Gradio interface may not function correctly.")

    print("Launching Gradio interface... Access it in your browser (usually at http://127.0.0.1:7860 or http://localhost:7860)")
    demo.launch()

    # Cleanup is mostly handled per-request for frame directories.
    # Persisted output videos are in TEMP_VIDEO_DIR.

    # Cleanup of parent temporary directories on exit could be added here if desired,
    # but typically Gradio or the OS handles broader temp file cleanup.
    # The script cleans up per-request frame directories. Video files in TEMP_VIDEO_DIR will persist.
    # For a production app, a more robust temporary file management strategy for videos might be needed. 