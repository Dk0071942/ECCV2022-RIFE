import shutil
import datetime
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

from rife_app.config import DEVICE, IMAGE_TMP_DIR, VIDEO_TMP_DIR
from rife_app.utils.framing import pil_to_tensor, pad_tensor_for_rife, save_tensor_as_image
from rife_app.utils.interpolation import generate_interpolated_frames
from rife_app.utils.disk_based_interpolation import DiskBasedInterpolator
from rife_app.utils.ffmpeg import run_ffmpeg_command

class ImageInterpolator:
    def __init__(self, model):
        self.model = model
        self.disk_based_interpolator = DiskBasedInterpolator(model, DEVICE)

    def interpolate(self, img0_pil: Image.Image, img1_pil: Image.Image, num_passes: int, fps: int, 
                   use_disk_based: bool = False):
        if img0_pil is None or img1_pil is None:
            return None, "Please upload both images."

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_op_dir = IMAGE_TMP_DIR / f"std_interp_{timestamp}"
        frames_dir = unique_op_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare tensors
            img0_tensor = pil_to_tensor(img0_pil, DEVICE)
            
            # Pad the first image and get its new dimensions
            img0_padded, (h, w) = pad_tensor_for_rife(img0_tensor)
            
            # Pad the second image, ensuring it matches the first one's padding exactly
            img1_tensor = pil_to_tensor(img1_pil, DEVICE)
            ph = img0_padded.shape[2]
            pw = img0_padded.shape[3]
            padding = (0, pw - img1_tensor.shape[3], 0, ph - img1_tensor.shape[2])
            img1_padded = F.pad(img1_tensor, padding)

            # Generate frames using selected method
            if use_disk_based:
                # Use disk-based interpolation with passes logic
                target_frames = (2 ** num_passes)  # Convert passes to frame count
                from rife_app.utils.disk_based_interpolation import disk_based_interpolate
                video_path, status_msg = disk_based_interpolate(
                    img0_padded, img1_padded, self.model, target_frames=target_frames, device=DEVICE
                )
                
                if video_path:
                    duration_seconds = target_frames / 25.0
                    # Ensure video_path is a string for Gradio
                    return str(video_path), f"Disk-based interpolation: {num_passes} passes → {target_frames} frames → {duration_seconds:.2f}s at 25 FPS. {status_msg}"
                else:
                    return None, f"Disk-based interpolation failed: {status_msg}"
                    
            else:
                # Use multiple passes of 2x interpolation between the two input frames
                # Each pass doubles the number of intermediate frames, creating longer 25 FPS videos
                current_frames = [img0_padded, img1_padded]  # Start with original pair
                
                for pass_num in range(num_passes):
                    new_frames = []
                    # Process each adjacent pair in current_frames
                    for i in range(len(current_frames) - 1):
                        frame_a = current_frames[i]
                        frame_b = current_frames[i + 1]
                        # Generate middle frame using exp=1 (optimal 2x interpolation)
                        middle_frames = generate_interpolated_frames(frame_a, frame_b, 1, self.model)
                        new_frames.append(frame_a)
                        new_frames.extend(middle_frames)
                    new_frames.append(current_frames[-1])  # Add final frame
                    current_frames = new_frames
                
                frame_tensors = current_frames
                total_frames = len(frame_tensors)
                duration_seconds = total_frames / 25.0
                interpolation_method = f"multiple passes ({num_passes} passes, {total_frames} frames, {duration_seconds:.2f}s at 25 FPS)"

            # Save frames
            for i, frame_tensor in enumerate(frame_tensors):
                frame_path = frames_dir / f'frame_{i:05d}.png'
                save_tensor_as_image(frame_tensor, frame_path, original_size=(h, w))

            # Create video with proper BT.709 color space metadata
            output_video_path = VIDEO_TMP_DIR / f"std_slomo_{timestamp}.mp4"
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-r', str(fps),
                '-i', frames_dir / 'frame_%05d.png',
                '-s', f'{w}x{h}',
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1',
                '-color_primaries', 'bt709',
                '-color_trc', 'bt709',
                '-colorspace', 'bt709',
                '-movflags', '+faststart',
                output_video_path
            ]
            
            success, msg = run_ffmpeg_command(ffmpeg_cmd, unique_op_dir)
            if not success:
                # Still handle FFmpeg errors gracefully
                if unique_op_dir.exists():
                    shutil.rmtree(unique_op_dir)
                return None, f"FFmpeg error: {msg}"

            # Cleanup on success
            if unique_op_dir.exists():
                shutil.rmtree(unique_op_dir)

            return str(output_video_path), f"Interpolation successful using {interpolation_method}. Generated {len(frame_tensors)} frames with optimal quality."
        except Exception as e:
            # Cleanup on error
            if unique_op_dir.exists():
                shutil.rmtree(unique_op_dir)
            return None, f"Interpolation error: {str(e)}" 