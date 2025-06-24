import shutil
import datetime
import math
from pathlib import Path
from PIL import Image
import cv2

from rife_app.config import DEVICE, CHAINED_TMP_DIR, VIDEO_TMP_DIR
from rife_app.utils.framing import get_video_info, pil_to_tensor, pad_tensor_for_rife, save_tensor_as_image
from rife_app.utils.interpolation import generate_interpolated_frames
from rife_app.utils.ffmpeg import run_ffmpeg_command, scale_and_pad_image

class ChainedInterpolator:
    def __init__(self, model):
        self.model = model

    def _generate_interpolation_segment(self, img_start_pil, img_end_pil, exp, fps, w, h, frames_dir, segment_path):
        img_start_tensor = pil_to_tensor(img_start_pil, DEVICE)
        img_end_tensor = pil_to_tensor(img_end_pil, DEVICE)

        img_start_padded, _ = pad_tensor_for_rife(img_start_tensor)
        img_end_padded, _ = pad_tensor_for_rife(img_end_tensor)
        
        frame_tensors = generate_interpolated_frames(img_start_padded, img_end_padded, exp, self.model)
        
        for i, frame_tensor in enumerate(frame_tensors):
            save_tensor_as_image(frame_tensor, frames_dir / f'frame_{i:05d}.png', original_size=(h, w))
            
        cmd = [
            'ffmpeg', '-y', '-r', str(fps), '-i', frames_dir / 'frame_%05d.png',
            '-s', f'{w}x{h}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', segment_path
        ]
        success, msg = run_ffmpeg_command(cmd)
        if not success:
            raise Exception(f"FFmpeg error for segment {segment_path.name}: {msg}")

    def interpolate(self, anchor_img_pil, middle_video_path, exp_value, interp_duration_seconds, final_fps):
        if not all([anchor_img_pil, middle_video_path]):
            raise Exception("Anchor image and middle video are required.")
        if interp_duration_seconds <= 0 or final_fps <= 0:
            raise Exception("Durations and FPS must be positive.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        op_dir = CHAINED_TMP_DIR / f"chained_{timestamp}"
        op_dir.mkdir(parents=True)
        
        try:
            # Get target resolution from middle video
            middle_info = get_video_info(Path(middle_video_path))
            if not middle_info or middle_info['width'] == 0:
                raise Exception("Could not read middle video properties.")
            target_w, target_h = middle_info['width'], middle_info['height']

            # Prepare directories
            temp_img_dir = op_dir / "temp_images"
            frames1_dir = op_dir / "frames1"
            frames2_dir = op_dir / "frames2"
            segments_dir = op_dir / "segments"
            for d in [temp_img_dir, frames1_dir, frames2_dir, segments_dir]:
                d.mkdir()

            # Prepare anchor image
            scaled_anchor_path = temp_img_dir / "anchor_scaled.png"
            anchor_img_pil.save(temp_img_dir / "anchor_orig.png")
            scale_and_pad_image(temp_img_dir / "anchor_orig.png", target_w, target_h, scaled_anchor_path)
            scaled_anchor_pil = Image.open(scaled_anchor_path)

            # Extract and prepare middle video frames
            cap = cv2.VideoCapture(middle_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, vid_first_bgr = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_info['frame_count'] - 1)
            _, vid_last_bgr = cap.read()
            cap.release()

            vid_first_pil = Image.fromarray(cv2.cvtColor(vid_first_bgr, cv2.COLOR_BGR2RGB))
            vid_last_pil = Image.fromarray(cv2.cvtColor(vid_last_bgr, cv2.COLOR_BGR2RGB))

            # Calculate interpolation FPS
            num_frames = (2**exp_value) + 1
            interp_fps = max(0.1, math.ceil(num_frames / interp_duration_seconds * 100) / 100.0)

            # Interpolation 1: Anchor -> Video Start
            interp1_path = segments_dir / "interp1.mp4"
            self._generate_interpolation_segment(scaled_anchor_pil, vid_first_pil, exp_value, interp_fps, target_w, target_h, frames1_dir, interp1_path)

            # Interpolation 2: Video End -> Anchor
            interp2_path = segments_dir / "interp2.mp4"
            self._generate_interpolation_segment(vid_last_pil, scaled_anchor_pil, exp_value, interp_fps, target_w, target_h, frames2_dir, interp2_path)

            # Prepare Middle Video
            middle_reencoded_path = segments_dir / "middle_reencoded.mp4"
            vf_filter = f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=1,pad=w={target_w}:h={target_h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black"
            cmd_middle = ['ffmpeg', '-y', '-i', middle_video_path, '-vf', vf_filter, '-r', str(final_fps), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', middle_reencoded_path]
            success, msg = run_ffmpeg_command(cmd_middle)
            if not success: raise Exception(f"FFmpeg for Middle video prep: {msg}")

            # Concatenate
            concat_list_path = op_dir / "concat_list.txt"
            with open(concat_list_path, 'w') as f:
                f.write(f"file '{interp1_path.resolve()}'\n")
                f.write(f"file '{middle_reencoded_path.resolve()}'\n")
                f.write(f"file '{interp2_path.resolve()}'\n")

            final_video_path = VIDEO_TMP_DIR / f"chained_output_{timestamp}.mp4"
            cmd_concat = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', '-r', str(final_fps), final_video_path]
            success, msg = run_ffmpeg_command(cmd_concat)
            if not success: raise Exception(f"FFmpeg for Concatenation: {msg}")
            
            return str(final_video_path), "Chained interpolation successful."
        
        except Exception as e:
            raise e
        finally:
            if op_dir.exists():
                shutil.rmtree(op_dir) 