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
        print("RIFE app will launch in degraded mode - interpolation features will be disabled.")
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

def handle_advanced_video_interpolation(
    video_path, 
    num_passes, 
    progress=gr.Progress(track_tqdm=True)
):
    """Handles video frame rate interpolation using multiple passes for optimal quality."""
    if not video_path:
        return None, "No video uploaded"
    
    if not video_interp:
        return None, "❌ RIFE model not loaded. This may be due to missing GPU drivers or model files. Check deployment logs for details."
    
    output_dir = Path("./temp_gradio/interpolated_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Video interpolation: Run multiple passes of 2x frame rate increase
        # This maintains video duration while increasing frame rate
        current_video_path = video_path
        
        for pass_num in range(num_passes):
            pass_output_dir = output_dir / f"pass_{pass_num + 1}"
            pass_output_dir.mkdir(exist_ok=True)
            
            # Each pass uses exp=1 (2x interpolation) to double frame rate
            current_video_path = main_interpolate(
                input_video_path=current_video_path,
                output_dir_path=str(pass_output_dir),
                exp=1,  # Always use exp=1 for optimal quality
                use_fp16=False,
            )
        
        final_multiplier = 2 ** num_passes
        return current_video_path, f"Video interpolated successfully through {num_passes} passes to {final_multiplier}x frame rate using optimal 2x interpolations.\nOutput: {current_video_path}"
        
    except Exception as e:
        raise gr.Error(str(e))

def update_memory_estimation(interpolation_method, num_passes, img0, img1):
    """Update memory usage estimation based on current parameters."""
    if img0 is None or img1 is None:
        return "Upload images to see memory estimation"
    
    try:
        # Estimate based on typical image size (assume 512x512 if no actual size available)
        mb_per_frame = (3 * 512 * 512 * 4) / (1024 * 1024)  # Rough estimate
        
        if interpolation_method == "disk_based":
            total_frames = (2 ** num_passes)
            duration_seconds = total_frames / 25.0
            return f"💾 Disk-Based: {num_passes} passes → {total_frames} frames → {duration_seconds:.2f}s at 25 FPS, Peak GPU memory: ~{mb_per_frame*2:.1f}MB (constant memory usage)"
        else:
            # Estimate for multiple passes mode - creates longer duration at 25 FPS
            total_frames = (2 ** num_passes)
            duration_seconds = total_frames / 25.0
            # Memory usage is constant per pass (only 2 frames in memory at once)
            peak_mb = mb_per_frame * 2  # Only 2 frames in memory per pass
            
            return f"⚡ Multiple Passes: {num_passes} passes → {total_frames} frames → {duration_seconds:.2f}s at 25 FPS, Peak GPU memory per pass: ~{peak_mb:.1f}MB"
            
    except Exception as e:
        return f"Estimation error: {str(e)}"

def handle_video_reencoding(video_path, force_frame_based_reencoding):
    """Simple video re-encoding handler with comprehensive input logging."""
    print("=" * 60)
    print("🎬 GRADIO HANDLER - Video Re-encoding Started")
    print(f"🔍 GRADIO INPUT TYPE: {type(video_path)}")
    print(f"🔍 GRADIO INPUT REPR: {repr(video_path)}")
    print(f"🔍 GRADIO INPUT BOOL: {bool(video_path)}")
    print(f"🔄 FORCE FRAME-BASED REENCODING: {force_frame_based_reencoding}")
    
    if hasattr(video_path, '__dict__'):
        print(f"🔍 GRADIO INPUT VARS: {vars(video_path)}")
    
    if not video_path:
        print("❌ GRADIO HANDLER - No video uploaded")
        return None, "No video uploaded", ""
    
    print("🔄 GRADIO HANDLER - Calling reencoder...")
    
    # Re-encoding call with force parameter
    reencoded_video_path, status_message = video_reencoder.reencode_video(video_path, force_frame_based_reencoding)
    
    print(f"✅ GRADIO HANDLER - Reencoder returned: {reencoded_video_path is not None}")
    print("=" * 60)
    
    # Get encoding info
    encoding_info = video_reencoder.get_info()
    
    return reencoded_video_path, status_message, encoding_info

def create_rife_ui():
    """Creates the Gradio UI for video and image interpolation."""
    # --- Gradio Interface ---
    gr.Markdown("# Video and Image Frame Interpolation")
    
    # Model status indicator
    model_status = "✅ RIFE model loaded successfully" if image_interp else "❌ RIFE model failed to load - features disabled"
    gr.Markdown(f"**Model Status**: {model_status}")
    
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
            gr.Markdown("### Create smooth transition video between two images")
            gr.Markdown("**Duration Control**: More passes create longer videos at fixed 25 FPS")
            gr.Markdown("*When using frames from Tab 1, the **end frame** becomes the **first image** and the **start frame** becomes the **second image**, creating a reverse interpolation.*")
            with gr.Row():
                img0_input_interpolation = gr.Image(type="pil", label="First Image (Source)")
                img1_input_interpolation = gr.Image(type="pil", label="Second Image (Target)")
            passes_slider = gr.Slider(minimum=1, maximum=6, value=2, step=1, label="Number of Passes", info="Each pass doubles video duration at 25 FPS (1=0.08s, 2=0.16s, 3=0.32s, 4=0.64s, 5=1.28s, 6=2.56s)")
            
            # Interpolation Mode Selection
            with gr.Row():
                interpolation_mode = gr.Radio(
                    choices=[
                        ("Standard (Recursive)", "standard"),
                        ("Disk-Based (Best Quality)", "disk_based")
                    ],
                    value="disk_based",
                    label="Interpolation Method",
                    info="Choose interpolation approach: Standard (fastest), Disk-based (best quality, no motion blur)"
                )
            # Method-specific controls
            with gr.Row(visible=True) as disk_based_controls:
                gr.Markdown("💡 **Disk-based mode**: Uses passes logic with constant memory usage. Same number of passes as standard mode, but stores frames on disk for memory efficiency.")
            
            # Memory usage estimation display
            memory_estimation_display = gr.Textbox(
                label="Memory Estimation",
                interactive=False,
                value="Upload images to see memory estimation",
                lines=2
            )
            
            # Fixed 25 FPS output - remove user control
            # fps_number = gr.Number(value=25, label="Output Video FPS (Fixed)", minimum=25, maximum=25, interactive=False)
            interpolate_button = gr.Button("Generate Interpolated Video", variant="primary")
            video_output_display = gr.Video(label="Output Video")
            interpolation_status_display = gr.Textbox(label="Status", interactive=False)

        # Tab 3: Chained Interpolation
        with gr.TabItem("3. Chained Video Interpolation"):
            gr.Markdown("### Chain multiple videos with smooth transitions")
            gr.Markdown("**Enhanced Method Selection**: Choose interpolation method based on your needs")
            gr.Markdown("**Duration Control**: More passes create longer transition segments at 25 FPS")
            gr.Markdown("Upload your videos in sequence from left to right:")
            with gr.Row():
                anchor_video_chained = gr.Video(label="Anchor Video (Start)")
                middle_video_chained = gr.Video(label="Middle Video")
                end_video_chained = gr.Video(label="End Video")
            
            with gr.Row():
                passes_chained_interp = gr.Slider(minimum=1, maximum=6, value=2, step=1, label="Number of Passes", info="Each pass doubles transition duration (2 passes = ~160ms, 6 passes = ~2.56s)")
                main_video_fps_chained = gr.Number(value=DEFAULT_FPS, label="Final FPS")
                
            # Interpolation Mode Selection - matching Tab 2 format
            interp_method_chained = gr.Radio(
                choices=[
                    ("Standard (Recursive)", "image_interpolation"),
                    ("Disk-Based (Best Quality)", "disk_based")
                ],
                value="disk_based",
                label="Interpolation Method",
                info="Choose interpolation approach: Standard (fastest), Disk-based (best quality, no motion blur)"
            )
            
            btn_run_chained_interp = gr.Button("Generate Chained Video", variant="primary")
            vid_output_chained_interp = gr.Video(label="Chained Output Video")
            status_chained_interp = gr.Textbox(label="Status", interactive=False)

        # Tab 4: Video Interpolation
        with gr.TabItem("4. Video Interpolation"):
            gr.Markdown("### Video Frame Rate Interpolation")
            gr.Markdown("**Frame Rate Control**: Increase video frame rate while keeping duration constant.")
            gr.Markdown("Interpolate video frames to increase FPS using multiple optimal 2x passes.")
            gr.Markdown("💡 **Frame Rate Control**: More passes increase video frame rate while keeping duration constant. Research shows multiple 2x passes produce sharper results than single high-factor interpolations.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input_tab4 = gr.Video(label="Upload Video for Interpolation")
                    
                    # Standard interpolation controls
                    with gr.Row():
                        video_passes_slider = gr.Slider(
                            minimum=1, maximum=4, value=1, step=1,
                            label="Number of Passes", 
                            info="Each pass doubles frame rate, keeping video duration (1=2x FPS, 2=4x FPS, 3=8x FPS, 4=16x FPS)"
                        )
                        
                    interpolate_video_button_tab4 = gr.Button("Interpolate Video", variant="primary")
                    
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
                    force_reencoding_checkbox = gr.Checkbox(
                        label="Use frame-based reencoding (extract frames)", 
                        value=True,
                        info="Check this to use frame extraction method for perfect color consistency. Unchecked = faster direct reencoding. Checked = slower frame-based reencoding with color analysis."
                    )
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
    # Toggle visibility of method-specific controls
    def update_method_controls(method):
        return gr.Row(visible=(method == "disk_based"))     # disk_based_controls
    
    interpolation_mode.change(
        update_method_controls,
        inputs=[interpolation_mode],
        outputs=[disk_based_controls]
    )
    
    # Update memory estimation when parameters change
    for input_component in [interpolation_mode, passes_slider, 
                           img0_input_interpolation, img1_input_interpolation]:
        input_component.change(
            update_memory_estimation,
            inputs=[interpolation_mode, passes_slider, 
                   img0_input_interpolation, img1_input_interpolation],
            outputs=[memory_estimation_display]
        )
    
    def handle_interpolation(img0, img1, num_passes, method):
        """Handle interpolation with different methods."""
        if not image_interp:
            return None, "❌ RIFE model not loaded. This may be due to missing GPU drivers or model files. Check deployment logs for details."
        
        # Convert method to boolean flags for backward compatibility
        use_disk_based = (method == "disk_based")
        
        return image_interp.interpolate(
            img0, img1, num_passes, 25,  # Fixed 25 FPS
            use_disk_based=use_disk_based
        )
    
    interpolate_button.click(
        handle_interpolation,
        inputs=[img0_input_interpolation, img1_input_interpolation, passes_slider, 
                interpolation_mode],
        outputs=[video_output_display, interpolation_status_display]
    )

    # Tab 3
    def handle_chained_interpolation(anchor_video, middle_video, end_video, passes, fps, method):
        """Handle chained interpolation with exact Tab 2 quality."""
        if not chained_interp:
            return None, "❌ RIFE model not loaded. This may be due to missing GPU drivers or model files. Check deployment logs for details."
        try:
            # Determine use_disk_based from method selection (matches Tab 2 logic)
            use_disk_based = (method == "disk_based")
            
            # Pass all parameters including use_disk_based for perfect Tab 2 quality
            return chained_interp.interpolate(
                anchor_video, middle_video, end_video, 
                passes, fps, method, 
                use_disk_based=use_disk_based  # ✅ Pass Tab 2 quality parameter
            )
        except Exception as e:
            return None, f"Chained interpolation failed: {str(e)}"
    
    btn_run_chained_interp.click(
        handle_chained_interpolation,
        inputs=[anchor_video_chained, middle_video_chained, end_video_chained, passes_chained_interp, main_video_fps_chained, interp_method_chained],
        outputs=[vid_output_chained_interp, status_chained_interp]
    )
    
    # Tab 4  
    interpolate_video_button_tab4.click(
        handle_advanced_video_interpolation,
        inputs=[video_input_tab4, video_passes_slider],
        outputs=[video_output_tab4, status_text_tab4]
    )
    
    # Tab 5: Video Re-encoding
    reencode_button.click(
        handle_video_reencoding,
        inputs=[video_input_reencoding, force_reencoding_checkbox],
        outputs=[video_output_reencoding, status_text_reencoding, encoding_info_display]
    )
    
    gr.Markdown("---")
    gr.Markdown("Ensure FFmpeg is installed and in your system's PATH. Model files should be in the `train_log` directory.")


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting RIFE Gradio Application")
    print(f"🐍 Python version: {sys.version}")
    import torch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🎯 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📁 Model directory: {os.path.abspath('./train_log')}")
    print(f"🛠️ FFmpeg available: {shutil.which('ffmpeg') is not None}")
    print("=" * 60)
    
    with gr.Blocks(title="Frame Interpolation Tool", theme=gr.themes.Soft()) as demo:
        create_rife_ui()
    
    # --- Authentication ---
    # Get credentials from environment variables with fallback to defaults
    # Set these environment variables in your deployment (e.g., Coolify):
    # - AUTH_USERNAME: Username for authentication
    # - AUTH_PASSWORD: Password for authentication
    AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "")
    AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "")

    # Only enable authentication if both username and password are provided
    if AUTH_USERNAME and AUTH_PASSWORD:
        auth_creds = (AUTH_USERNAME, AUTH_PASSWORD)
        print(f"Authentication enabled for user: {AUTH_USERNAME}")
    else:
        auth_creds = None
        print("Authentication disabled - no credentials configured")
    # --- End Authentication ---
    
    print("Launching Gradio interface...")
    demo.launch(
        server_port=7860,
        server_name="0.0.0.0",
        inbrowser=False,
        auth=auth_creds, # Pass credentials tuple to enable authentication
    ) 