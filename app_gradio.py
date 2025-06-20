import gradio as gr
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
import shutil
import datetime
import math
import imageio
from tqdm import tqdm

# Import from our refactored, modular scripts
from run_interpolation import main_interpolate
from rife_utils import (
    get_rife_model, 
    get_video_properties,
    cv2_frame_reader,
    transfer_audio_ffmpeg,
    make_rife_inference,
    run_ffmpeg_command
)

# --- Configuration & Globals ---
TEMP_VIDEO_DIR = "temp_gradio_videos"
TEMP_CHAINED_OP_DIR = "temp_chained_operations"
DEFAULT_FPS = 24

os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
os.makedirs(TEMP_CHAINED_OP_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# --- Core Functions ---

def get_model():
    """Alias for the model loader in rife_utils."""
    return get_rife_model()

def _generate_and_save_interpolated_frames(img0_pil, img1_pil, exp_value, target_frame_directory, model_instance):
    """Generates and saves interpolated frames between two PIL images."""
    os.makedirs(target_frame_directory, exist_ok=True)
    try:
        img0_np = np.array(img0_pil.convert("RGB"))
        img1_np = np.array(img1_pil.convert("RGB"))

        img0_tensor = torch.from_numpy(img0_np.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.
        img1_tensor = torch.from_numpy(img1_np.transpose(2, 0, 1)).to(device).unsqueeze(0).float() / 255.

        _n, _c, h_orig, w_orig = img0_tensor.shape
        
        ph = ((h_orig - 1) // 32 + 1) * 32
        pw = ((w_orig - 1) // 32 + 1) * 32
        padding = (0, pw - w_orig, 0, ph - h_orig)
        img0_padded = F.pad(img0_tensor, padding)
        img1_padded = F.pad(img1_tensor, padding)

    except Exception as e:
        return False, f"Preprocessing for RIFE failed: {e}", [], (0, 0)

    interpolated_tensors = make_rife_inference(img0_padded, img1_padded, exp_value, model_instance, model_scale_factor_for_inference=1.0)
    
    output_tensors = [img0_padded] + interpolated_tensors + [img1_padded]

    try:
        saved_frame_paths = []
        for i, frame_tensor in enumerate(output_tensors):
            img_np_cropped = frame_tensor[0, :, :h_orig, :w_orig].cpu().numpy().transpose(1, 2, 0)
            img_uint8 = (img_np_cropped * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            frame_path = os.path.join(target_frame_directory, f'frame_{i:05d}.png')
            cv2.imwrite(frame_path, img_bgr)
            saved_frame_paths.append(frame_path)
        return True, "Frames generated successfully.", saved_frame_paths, (w_orig, h_orig)
    except Exception as e:
        return False, f"Frame saving error: {e}", [], (w_orig, h_orig)

def interpolate_uploaded_video(
    input_video_path, 
    interpolation_exp,
    model_inference_scale_factor,
    output_resolution_scale_factor,
    target_fps_override,
    use_fp16,
    progress=gr.Progress(track_tqdm=True)
):
    if not input_video_path:
        raise gr.Error("Input video not provided.")
    
    try:
        output_video = main_interpolate(
            input_video_path=input_video_path,
            output_dir_path=TEMP_VIDEO_DIR,
            exp=interpolation_exp,
            use_fp16=use_fp16,
            model_inference_scale_factor=model_inference_scale_factor,
            output_resolution_scale_factor=output_resolution_scale_factor,
            target_fps_override=target_fps_override,
            progress_tqdm=progress,
        )
        
        status_message = f"Interpolation successful. Video saved at {output_video}"
        return output_video, status_message

    except Exception as e:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Video interpolation failed: {type(e).__name__} - {str(e)}")

def get_video_info(video_file_path):
    if video_file_path is None:
        info_text = "Video not loaded. Please upload a video."
        start_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        end_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        return info_text, start_update, end_update

    try:
        width, height, fps, frame_count = get_video_properties(video_file_path)

        if frame_count > 0:
            info_text = f"Video Info: {frame_count} frames, {fps:.2f} FPS, {width}x{height}."
            start_update = gr.update(minimum=1, maximum=frame_count, value=1, interactive=True)
            end_update = gr.update(minimum=1, maximum=frame_count, value=frame_count, interactive=True)
        else:
            info_text = f"Video Info: 0 frames found. Cannot extract."
            start_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
            end_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        
        return info_text, start_update, end_update
    except Exception as e:
        info_text = f"Error reading video info: {str(e)}"
        start_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        end_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        return info_text, start_update, end_update

def extract_frames_from_video(video_file_path, start_frame_one_indexed, end_frame_one_indexed):
    if video_file_path is None:
        raise gr.Error("Please upload a video first.")

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise gr.Error(f"Error: Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not (1 <= start_frame_one_indexed <= total_frames and 1 <= end_frame_one_indexed <= total_frames):
        cap.release()
        raise gr.Error(f"Frame numbers are out of bounds (1-{total_frames}).")
    if start_frame_one_indexed >= end_frame_one_indexed:
        cap.release()
        raise gr.Error(f"Start frame must be less than end frame.")

    start_0_idx, end_0_idx = start_frame_one_indexed - 1, end_frame_one_indexed - 1
    pil_start, pil_end = None, None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_0_idx)
        ret_start, frame_start = cap.read()
        if ret_start:
            pil_start = Image.fromarray(cv2.cvtColor(frame_start, cv2.COLOR_BGR2RGB))

        cap.set(cv2.CAP_PROP_POS_FRAMES, end_0_idx)
        ret_end, frame_end = cap.read()
        if ret_end:
            pil_end = Image.fromarray(cv2.cvtColor(frame_end, cv2.COLOR_BGR2RGB))
        
        if not ret_start or not ret_end:
             raise gr.Error("Could not read start or end frame.")
        
    except Exception as e:
        raise gr.Error(f"Error during frame extraction: {str(e)}")
    finally:
        cap.release()

    return pil_start, pil_end, f"Extracted frames {start_frame_one_indexed} and {end_frame_one_indexed}."

def create_standard_interpolated_video(img0_pil, img1_pil, exp_value, fps):
    if not all([img0_pil, img1_pil]):
        raise gr.Error("Both start and end images are required.")
    
    unique_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    op_dir = os.path.join(TEMP_CHAINED_OP_DIR, f"standard_{unique_id}")

    try:
        model_instance = get_model()
        success, message, saved_paths, (w,h) = _generate_and_save_interpolated_frames(
            img0_pil, img1_pil, exp_value, op_dir, model_instance
        )

        if not success:
            raise gr.Error(f"Frame generation failed: {message}")

        output_video_path = os.path.join(TEMP_VIDEO_DIR, f"std_slomo_{unique_id}.mp4")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-r', str(fps), '-i', os.path.join(op_dir, 'frame_%05d.png'),
            '-s', f'{w}x{h}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output_video_path
        ]
        
        success, msg = run_ffmpeg_command(ffmpeg_cmd, op_dir)
        if not success:
            raise gr.Error(f"FFmpeg error (standard): {msg}")

        if os.path.exists(op_dir):
            shutil.rmtree(op_dir)
            
        return output_video_path, "Standard interpolation successful."

    except Exception as e:
        if os.path.exists(op_dir):
            shutil.rmtree(op_dir)
        if isinstance(e, gr.Error):
            raise e
        raise gr.Error(f"Video Creation failed: {e}")

def get_middle_video_info(middle_video_path):
    if middle_video_path is None:
        return "Middle video not loaded.", gr.update(value=DEFAULT_FPS)
    try:
        fps = get_video_properties(middle_video_path)[2]
        fps_display_val = round(fps) if fps >= 1 else round(fps, 2)
        return f"Middle Video FPS: {fps:.2f}", gr.update(value=fps_display_val)
    except Exception as e:
        return f"Error reading middle video: {str(e)}", gr.update(value=DEFAULT_FPS)

def create_chained_interpolated_video(anchor_img_pil, middle_video_path, exp_value, interp_duration_seconds, main_video_final_fps):
    if not all([anchor_img_pil, middle_video_path]):
        raise gr.Error("Anchor image and middle video are required.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    operation_dir = os.path.join(TEMP_CHAINED_OP_DIR, f"chained_{timestamp}")
    video_segments_dir = os.path.join(operation_dir, "video_segments")
    os.makedirs(video_segments_dir, exist_ok=True)

    try:
        target_w, target_h, middle_original_fps, _ = get_video_properties(middle_video_path)
        
        videogen = cv2_frame_reader(middle_video_path)
        first_frame_np = next(videogen)
        last_frame_np = first_frame_np
        for frame in videogen:
            last_frame_np = frame
            
        first_frame_pil = Image.fromarray(first_frame_np)
        last_frame_pil = Image.fromarray(last_frame_np)

        num_frames_per_interp_segment = (2**exp_value) + 1
        interp_segment_fps = max(1, math.ceil(num_frames_per_interp_segment / interp_duration_seconds))

        interp1_path, _ = create_standard_interpolated_video(anchor_img_pil, first_frame_pil, exp_value, interp_segment_fps)
        interp2_path, _ = create_standard_interpolated_video(last_frame_pil, anchor_img_pil, exp_value, interp_segment_fps)
        
        middle_reencoded_path = os.path.join(video_segments_dir, "middle_reencoded.mp4")
        cmd_middle = ['ffmpeg', '-y', '-i', middle_video_path, '-r', str(main_video_final_fps), '-c:v', 'libx264', '-an', middle_reencoded_path]
        run_ffmpeg_command(cmd_middle)

        concat_list_path = os.path.join(operation_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            f.write(f"file '{os.path.abspath(interp1_path)}'\n")
            f.write(f"file '{os.path.abspath(middle_reencoded_path)}'\n")
            f.write(f"file '{os.path.abspath(interp2_path)}'\n")

        final_video_path = os.path.join(TEMP_VIDEO_DIR, f"chained_output_{timestamp}.mp4")
        cmd_concat = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', final_video_path]
        success, msg = run_ffmpeg_command(cmd_concat, operation_dir)
        if not success: raise gr.Error(f"Concatenation failed: {msg}")

        transfer_audio_ffmpeg(middle_video_path, final_video_path, operation_dir)

        return final_video_path, "Chained interpolation successful."
    except Exception as e:
        raise gr.Error(f"Chained interpolation failed: {e}")
    finally:
        if os.path.exists(operation_dir):
            shutil.rmtree(operation_dir)

# --- Gradio UI ---
if __name__ == '__main__':
    try:
        get_model()
        print("RIFE model loaded successfully for Gradio app.")
    except Exception as e:
        print(f"FATAL: Failed to load RIFE model at startup: {e}")
        exit()

    with gr.Blocks(title="RIFE Video and Image Interpolation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# RIFE Video and Image Frame Interpolation")

        with gr.Tabs():
            with gr.TabItem("1. Select Frames from Video (Optional)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video")
                        video_info_display = gr.Textbox(label="Video Information", interactive=False)
                        start_frame_num_input = gr.Number(label="Start Frame", value=1, precision=0, interactive=False)
                        end_frame_num_input = gr.Number(label="End Frame", value=1, precision=0, interactive=False)
                        extract_button = gr.Button("Extract & Use Frames", variant="secondary")
                        extraction_status_display = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("### Extracted Frames:")
                        extracted_start_img_display = gr.Image(label="Start Frame", type="pil", interactive=False)
                        extracted_end_img_display = gr.Image(label="End Frame", type="pil", interactive=False)
            
            with gr.TabItem("2. Interpolate Between Images"):
                with gr.Row():
                    img0_input_interpolation = gr.Image(type="pil", label="First Image")
                    img1_input_interpolation = gr.Image(type="pil", label="Second Image")
                exp_slider = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Interpolation Factor (2^exp)")
                fps_number = gr.Number(value=24, label="Output Video FPS")
                interpolate_button = gr.Button("Generate Video", variant="primary")
                video_output_display = gr.Video(label="Output Video")
                interpolation_status_display = gr.Textbox(label="Status", interactive=False)

            with gr.TabItem("3. Chained Video Interpolation"):
                with gr.Row():
                    anchor_img_chained = gr.Image(type="pil", label="Anchor Image (Start/End)")
                    middle_video_chained = gr.Video(label="Middle Video")
                exp_chained_interp = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Interpolation Factor")
                interp_duration_chained = gr.Number(value=2.0, label="Interpolation Duration (s)")
                main_video_fps_chained = gr.Number(value=DEFAULT_FPS, label="Final FPS")
                btn_run_chained_interp = gr.Button("Generate Chained Video", variant="primary")
                vid_output_chained_interp = gr.Video(label="Chained Output")
                status_chained_interp = gr.Textbox(label="Status", interactive=False)

            with gr.TabItem("4. Interpolate Video FPS"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_tab4 = gr.Video(label="Upload Video")
                        interp_exp_slider_tab4 = gr.Slider(minimum=0, maximum=4, value=1, step=1, label="Interpolation Multiplier (2^exp)")
                        model_scale_slider_tab4 = gr.Dropdown(label="Model Inference Scale", choices=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)
                        output_res_scale_slider_tab4 = gr.Slider(minimum=0.25, maximum=2.0, value=1.0, step=0.05, label="Output Resolution Scale")
                        target_fps_number_tab4 = gr.Number(label="Target FPS (Optional)", value=None)
                        fp16_checkbox_tab4 = gr.Checkbox(label="Use FP16 Inference", value=False)
                        interpolate_video_button_tab4 = gr.Button("Interpolate Video FPS", variant="primary")
                    with gr.Column(scale=1):
                        video_output_tab4 = gr.Video(label="Interpolated Video")
                        status_text_tab4 = gr.Textbox(label="Status", interactive=False, lines=10)

        # --- Event Handlers ---
        video_input.upload(fn=get_video_info, inputs=[video_input], outputs=[video_info_display, start_frame_num_input, end_frame_num_input])
        extract_button.click(fn=extract_frames_from_video, inputs=[video_input, start_frame_num_input, end_frame_num_input], outputs=[extracted_start_img_display, extracted_end_img_display, extraction_status_display])
        
        # Link extracted frames to the next tab
        extract_button.click(lambda x,y: (x,y), inputs=[extracted_start_img_display, extracted_end_img_display], outputs=[img0_input_interpolation, img1_input_interpolation])

        interpolate_button.click(fn=create_standard_interpolated_video, inputs=[img0_input_interpolation, img1_input_interpolation, exp_slider, fps_number], outputs=[video_output_display, interpolation_status_display])
        
        middle_video_chained.upload(fn=get_middle_video_info, inputs=[middle_video_chained], outputs=[status_chained_interp, main_video_fps_chained])
        btn_run_chained_interp.click(fn=create_chained_interpolated_video, inputs=[anchor_img_chained, middle_video_chained, exp_chained_interp, interp_duration_chained, main_video_fps_chained], outputs=[vid_output_chained_interp, status_chained_interp])
        
        interpolate_video_button_tab4.click(fn=interpolate_uploaded_video, inputs=[video_input_tab4, interp_exp_slider_tab4, model_scale_slider_tab4, output_res_scale_slider_tab4, target_fps_number_tab4, fp16_checkbox_tab4], outputs=[video_output_tab4, status_text_tab4])

    print("Launching Gradio interface... Access it in your browser (usually at http://127.0.0.1:7860)")
    demo.launch()