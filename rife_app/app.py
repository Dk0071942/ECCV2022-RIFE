import gradio as gr
import shutil
from pathlib import Path

# Setup sys.path for relative imports
import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rife_app.config import DEFAULT_FPS, DEVICE
from rife_app.models.loader import get_model, setup_torch_device
from rife_app.services.image_interpolator import ImageInterpolator
from rife_app.services.video_interpolator import VideoInterpolator
from rife_app.services.chained import ChainedInterpolator
from rife_app.services.simple_reencoder import SimpleVideoReencoder
from rife_app.utils.framing import get_video_info, extract_frames
from rife_app.run_interpolation import main_interpolate

# --- App Setup ---
def initialize_app():
    """Initializes model and services."""
    setup_torch_device()
    try:
        model = get_model()
        print("RIFE model loaded successfully for Gradio app.")
    except RuntimeError as e:
        print(f"Failed to load RIFE model at startup: {e}")
        model = None
    
    # Pre-check for FFmpeg
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg was not found. Video generation will fail.")

    # Instantiate services
    image_interpolator = ImageInterpolator(model) if model else None
    video_interpolator = VideoInterpolator(model, DEVICE) if model else None
    chained_interpolator = ChainedInterpolator(model) if model else None
    video_reencoder = SimpleVideoReencoder()  # Simple video re-encoder
    
    return image_interpolator, video_interpolator, chained_interpolator, video_reencoder

image_interp, video_interp, chained_interp, video_reencoder = initialize_app()

# --- UI Helper Functions ---
def handle_video_upload(video_file_path):
    if video_file_path is None:
        return "Video not loaded.", gr.update(minimum=1, maximum=1, value=1, interactive=False), gr.update(minimum=1, maximum=1, value=1, interactive=False)
    
    info = get_video_info(Path(video_file_path))
    if info and info["frame_count"] > 0:
        total_frames = info["frame_count"]
        info_text = f"Video Info: {total_frames} frames, {info['fps']:.2f} FPS."
        start_update = gr.update(minimum=1, maximum=total_frames, value=1, interactive=True)
        end_val = total_frames if total_frames > 1 else 1
        end_update = gr.update(minimum=1, maximum=total_frames, value=end_val, interactive=True)
    else:
        info_text = "Could not read video info or video has 0 frames."
        start_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        end_update = gr.update(minimum=1, maximum=1, value=1, interactive=False)
        
    return info_text, start_update, end_update

def handle_frame_extraction(video_file_path, start_str, end_str):
    if not video_file_path:
        raise gr.Error("Please upload a video first.")
    try:
        start_frame, end_frame = int(start_str), int(end_str)
    except ValueError:
        raise gr.Error("Frame numbers must be integers.")

    info = get_video_info(Path(video_file_path))
    if not info or not (1 <= start_frame <= info['frame_count']) or not (1 <= end_frame <= info['frame_count']):
         raise gr.Error("Frame numbers are out of bounds.")
    if start_frame >= end_frame:
        raise gr.Error("Start frame must be strictly less than end frame.")

    img_start, img_end = extract_frames(Path(video_file_path), start_frame, end_frame)
    if not img_start or not img_end:
        raise gr.Error("Failed to extract frames from video.")
        
    # Return frames in reversed order for Tab 2 interpolation:
    # end frame as first image, start frame as second image
    return img_start, img_end, img_end, img_start, f"Extracted frames {start_frame} and {end_frame}. Tab 2 will interpolate from frame {end_frame} to frame {start_frame}."

def handle_simple_interpolation(video_path, progress=gr.Progress(track_tqdm=True)):
    """Handles the video interpolation process, including file management."""
    if not video_path:
        return None, "No video uploaded"
    
    output_dir = Path("./temp_gradio/interpolated_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use the simple main_interpolate function for 2x FPS interpolation
        interpolated_video_path = main_interpolate(
            input_video_path=video_path,
            output_dir_path=str(output_dir),
            exp=1,  # 2x interpolation
            use_fp16=False,
        )
        
        return interpolated_video_path, f"Video interpolated successfully to 2x FPS.\nOutput: {interpolated_video_path}"
        
    except Exception as e:
        raise gr.Error(str(e))

def handle_video_reencoding(video_path):
    """Simple video re-encoding handler with comprehensive input logging."""
    print("=" * 60)
    print("🎬 GRADIO HANDLER - Video Re-encoding Started")
    print(f"🔍 GRADIO INPUT TYPE: {type(video_path)}")
    print(f"🔍 GRADIO INPUT REPR: {repr(video_path)}")
    print(f"🔍 GRADIO INPUT BOOL: {bool(video_path)}")
    
    if hasattr(video_path, '__dict__'):
        print(f"🔍 GRADIO INPUT VARS: {vars(video_path)}")
    
    if not video_path:
        print("❌ GRADIO HANDLER - No video uploaded")
        return None, "No video uploaded", ""
    
    print("🔄 GRADIO HANDLER - Calling reencoder...")
    
    # Simple re-encoding call
    reencoded_video_path, status_message = video_reencoder.reencode_video(video_path)
    
    print(f"✅ GRADIO HANDLER - Reencoder returned: {reencoded_video_path is not None}")
    print("=" * 60)
    
    # Get encoding info
    encoding_info = video_reencoder.get_info()
    
    return reencoded_video_path, status_message, encoding_info

def create_rife_ui():
    """Creates the Gradio UI for RIFE interpolation."""
    # --- Gradio Interface ---
    gr.Markdown("# RIFE Video and Image Frame Interpolation")
    
    with gr.Tabs():
        # Tab 1: Frame Extraction
        with gr.TabItem("1. Select Frames from Video"):
            # ... (UI definition)
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video")
                    video_info_display = gr.Textbox(label="Video Information", interactive=False)
                    start_frame_num_input = gr.Number(label="Start Frame (1-indexed)", value=1, minimum=1, maximum=1, precision=0, interactive=False)
                    end_frame_num_input = gr.Number(label="End Frame (1-indexed)", value=1, minimum=1, maximum=1, precision=0, interactive=False)
                    extract_button = gr.Button("Extract & Use Frames", variant="secondary")
                    extraction_status_display = gr.Textbox(label="Extraction Status", interactive=False, show_label=False)
                with gr.Column(scale=1):
                    gr.Markdown("### Extracted Frames:")
                    extracted_start_img_display = gr.Image(label="Start Frame Preview", type="pil", interactive=False)
                    extracted_end_img_display = gr.Image(label="End Frame Preview", type="pil", interactive=False)
                    gr.Markdown("*Note: These frames will be automatically loaded in Tab 2 with reversed order for interpolation.*")

        # Tab 2: Image Interpolation
        with gr.TabItem("2. Interpolate Between Images"):
            gr.Markdown("### Interpolate between two images")
            gr.Markdown("*When using frames from Tab 1, the **end frame** becomes the **first image** and the **start frame** becomes the **second image**, creating a reverse interpolation.*")
            with gr.Row():
                img0_input_interpolation = gr.Image(type="pil", label="First Image (Source)")
                img1_input_interpolation = gr.Image(type="pil", label="Second Image (Target)")
            exp_slider = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Interpolation Factor (exp)")
            fps_number = gr.Number(value=DEFAULT_FPS, label="Output Video FPS", minimum=1)
            interpolate_button = gr.Button("Generate Interpolated Video", variant="primary")
            video_output_display = gr.Video(label="Output Video")
            interpolation_status_display = gr.Textbox(label="Status", interactive=False)

        # Tab 3: Chained Interpolation
        with gr.TabItem("3. Chained Video Interpolation"):
            # ... (UI definition)
            with gr.Row():
                anchor_img_chained = gr.Image(type="pil", label="Anchor Image (Start/End)")
                middle_video_chained = gr.Video(label="Middle Video")
            exp_chained_interp = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Interpolation Factor (exp)")
            interp_duration_chained = gr.Number(value=2.0, minimum=0.1, label="Interpolation Duration (s)")
            main_video_fps_chained = gr.Number(value=DEFAULT_FPS, label="Final FPS")
            btn_run_chained_interp = gr.Button("Generate Chained Video", variant="primary")
            vid_output_chained_interp = gr.Video(label="Chained Output Video")
            status_chained_interp = gr.Textbox(label="Status", interactive=False)

        # Tab 4: Simple Video Interpolation (2x FPS)
        with gr.TabItem("4. Simple Video Interpolation (2x FPS)"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input_tab4 = gr.Video(label="Upload Video for 2x FPS Interpolation")
                    interpolate_video_button_tab4 = gr.Button("Interpolate Video (2x FPS)", variant="primary")
                with gr.Column(scale=1):
                    video_output_tab4 = gr.Video(label="Interpolated Video Output")
                    status_text_tab4 = gr.Textbox(label="Processing Status", interactive=False, lines=10)

        # Tab 5: Video Re-encoding
        with gr.TabItem("5. Video Re-encoding"):
            gr.Markdown("## 🎥 Professional Video Re-encoding")
            gr.Markdown("Re-encode videos using standardized high-quality parameters for optimal compatibility and quality.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    video_input_reencoding = gr.Video(label="Upload Video to Re-encode")
                    reencode_button = gr.Button("🔄 Re-encode Video", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    video_output_reencoding = gr.Video(label="Re-encoded Video")
                    
            with gr.Row():
                with gr.Column(scale=1):
                    status_text_reencoding = gr.Textbox(
                        label="Processing Status", 
                        interactive=False, 
                        lines=8,
                        placeholder="Upload a video and click 'Re-encode Video' to start..."
                    )
                with gr.Column(scale=1):
                    encoding_info_display = gr.Textbox(
                        label="Encoding Information", 
                        interactive=False, 
                        lines=8,
                        value=SimpleVideoReencoder().get_info()
                    )

    # --- Event Handlers ---
    # Tab 1
    video_input.upload(handle_video_upload, inputs=[video_input], outputs=[video_info_display, start_frame_num_input, end_frame_num_input])
    video_input.clear(handle_video_upload, inputs=[video_input], outputs=[video_info_display, start_frame_num_input, end_frame_num_input])
    extract_button.click(
        handle_frame_extraction,
        inputs=[video_input, start_frame_num_input, end_frame_num_input],
        outputs=[extracted_start_img_display, extracted_end_img_display, img0_input_interpolation, img1_input_interpolation, extraction_status_display]
    )

    # Tab 2
    interpolate_button.click(
        lambda *args: image_interp.interpolate(*args) if image_interp else (None, "Model not loaded"),
        inputs=[img0_input_interpolation, img1_input_interpolation, exp_slider, fps_number],
        outputs=[video_output_display, interpolation_status_display]
    )

    # Tab 3
    btn_run_chained_interp.click(
        lambda *args: chained_interp.interpolate(*args) if chained_interp else (None, "Model not loaded"),
        inputs=[anchor_img_chained, middle_video_chained, exp_chained_interp, interp_duration_chained, main_video_fps_chained],
        outputs=[vid_output_chained_interp, status_chained_interp]
    )
    
    # Tab 4
    interpolate_video_button_tab4.click(
        handle_simple_interpolation,
        inputs=[video_input_tab4],
        outputs=[video_output_tab4, status_text_tab4]
    )
    
    # Tab 5: Video Re-encoding
    reencode_button.click(
        handle_video_reencoding,
        inputs=[video_input_reencoding],
        outputs=[video_output_reencoding, status_text_reencoding, encoding_info_display]
    )
    
    gr.Markdown("---")
    gr.Markdown("Ensure FFmpeg is installed and in your system's PATH. Model files should be in the `train_log` directory.")


if __name__ == '__main__':
    with gr.Blocks(title="RIFE Interpolation", theme=gr.themes.Soft()) as demo:
        create_rife_ui()
    
    print("Launching Gradio interface...")
    demo.launch(
        server_port=7860,
        server_name="0.0.0.0",
        inbrowser=False,
    ) 