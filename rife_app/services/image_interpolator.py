import shutil
import datetime
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

from rife_app.config import DEVICE, IMAGE_TMP_DIR, VIDEO_TMP_DIR
from rife_app.utils.framing import pil_to_tensor, pad_tensor_for_rife, save_tensor_as_image
from rife_app.utils.interpolation import generate_interpolated_frames
from rife_app.utils.ffmpeg import run_ffmpeg_command

class ImageInterpolator:
    def __init__(self, model):
        self.model = model

    def interpolate(self, img0_pil: Image.Image, img1_pil: Image.Image, exp_value: int, fps: int):
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

            # Generate frames
            frame_tensors = generate_interpolated_frames(img0_padded, img1_padded, exp_value, self.model)

            # Save frames
            for i, frame_tensor in enumerate(frame_tensors):
                frame_path = frames_dir / f'frame_{i:05d}.png'
                save_tensor_as_image(frame_tensor, frame_path, original_size=(h, w))

            # Create video
            output_video_path = VIDEO_TMP_DIR / f"std_slomo_{timestamp}.mp4"
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-r', str(fps), '-i', frames_dir / 'frame_%05d.png',
                '-s', f'{w}x{h}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
                '-movflags', '+faststart', output_video_path
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

            return output_video_path, "Standard interpolation successful."
        finally:
            # This ensures cleanup even if other unexpected errors occur
            if unique_op_dir.exists():
                shutil.rmtree(unique_op_dir) 