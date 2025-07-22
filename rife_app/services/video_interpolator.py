import torch
import shutil
import datetime
import cv2
from tqdm import tqdm
from pathlib import Path
from torch.nn import functional as F

from rife_app.config import VIDEO_TMP_DIR
from rife_app.utils.ffmpeg import run_ffmpeg_command, transfer_audio
from rife_app.utils.interpolation import recursive_interpolate_video_frames

def cv2_frame_reader(video_path: str):
    """A generator to read video frames using OpenCV, yielding RGB frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file at {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

class VideoInterpolator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _pad_image_for_video(self, img_tensor, model_scale_factor, use_fp16):
        _n, _c, h_orig, w_orig = img_tensor.shape
        tmp = max(32, int(32 / model_scale_factor))
        ph = ((h_orig - 1) // tmp + 1) * tmp
        pw = ((w_orig - 1) // tmp + 1) * tmp
        padding = (0, pw - w_orig, 0, ph - h_orig)
        
        padded_tensor = F.pad(img_tensor, padding)
        return padded_tensor.half() if use_fp16 else padded_tensor

    def _save_frame(self, tensor, path, output_h, output_w):
        tensor_slice = tensor[0, :, :output_h, :output_w]
        np_array = (tensor_slice.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
        # Assuming skvideo provides RGB, we save RGB. cv2 expects BGR, so we'd need conversion if saving with it.
        # Let's use skvideo.io.vwrite which expects RGB. Or stick to cv2 and convert.
        # For simplicity, using a library that directly writes numpy arrays is good.
        # But ffmpeg will read from disk, so format must be consistent. PNG is lossless.
        # The original used cv2.imwrite with a cvtColor. Let's replicate that for consistency.
        cv2.imwrite(str(path), cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR))

    def interpolate(
        self,
        input_video_path: str,
        interpolation_exp: int,
        model_scale_factor: float,
        output_res_scale_factor: float,
        target_fps_override: int,
        use_fp16: bool,
        progress # Gradio progress object
    ):
        if not input_video_path:
            raise Exception("Input video not provided.") # Gradio will catch this
        
        if use_fp16 and not torch.cuda.is_available():
            raise Exception("FP16 selected, but CUDA is not available.")

        input_path = Path(input_video_path)
        original_torch_dtype = torch.get_default_dtype()
        operation_dir = None

        try:
            if use_fp16:
                torch.set_default_dtype(torch.float16)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            operation_dir = VIDEO_TMP_DIR / f"vid_interp_op_{timestamp}"
            frames_output_dir = operation_dir / "interpolated_frames"
            frames_output_dir.mkdir(parents=True, exist_ok=True)
            
            final_output_video_path = VIDEO_TMP_DIR / f"interpolated_video_{timestamp}.mp4"

            # Using cv2 to get metadata to avoid skvideo's numpy.float issue
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {input_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if total_frames == 0:
                raise Exception("Input video contains zero frames.")

            output_w = max(1, int(input_w * output_res_scale_factor))
            output_h = max(1, int(input_h * output_res_scale_factor))
            
            # Calculate final FPS
            final_fps = target_fps_override if target_fps_override else (original_fps * (2**interpolation_exp))
            interpolation_info = f"Recursive: exp={interpolation_exp}"
            
            status_updates = [
                f"Input: {total_frames} frames, {original_fps:.2f} FPS, {input_w}x{input_h}",
                f"Output: Model Scale: {model_scale_factor}x, Res Scale: {output_res_scale_factor}x -> {output_w}x{output_h}",
                f"Interpolation: {interpolation_info}",
                f"Target FPS: {final_fps:.2f}",
                f"FP16: {'Enabled' if use_fp16 else 'Disabled'}"
            ]

            videogen = cv2_frame_reader(str(input_path))
            last_frame_np = next(videogen).copy()
            
            last_frame_tensor = torch.from_numpy(last_frame_np.transpose(2, 0, 1)).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            
            # Upscale if too small, then apply resolution scaling
            h_orig, w_orig = last_frame_tensor.shape[2], last_frame_tensor.shape[3]
            if h_orig < 512 or w_orig < 512:
                target_h = max(h_orig, 512)
                target_w = max(w_orig, 512)
                last_frame_tensor = F.interpolate(last_frame_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

            if output_res_scale_factor != 1.0:
                last_frame_tensor = F.interpolate(last_frame_tensor, size=(output_h, output_w), mode='bilinear', align_corners=False)
            
            I1_padded = self._pad_image_for_video(last_frame_tensor, model_scale_factor, use_fp16)
            
            self._save_frame(I1_padded, frames_output_dir / "frame_0000000.png", output_h, output_w)
            saved_frame_count = 1

            pbar = tqdm(total=total_frames - 1, desc="Interpolating Video Frames")
            progress(0, desc="Starting...")

            for frame_idx, current_frame_np_orig in enumerate(videogen, start=1):
                current_frame_np = current_frame_np_orig.copy()
                I0_padded = I1_padded

                current_frame_tensor = torch.from_numpy(current_frame_np.transpose(2, 0, 1)).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
                
                # Upscale if too small, then apply resolution scaling
                h_orig, w_orig = current_frame_tensor.shape[2], current_frame_tensor.shape[3]
                if h_orig < 512 or w_orig < 512:
                    target_h = max(h_orig, 512)
                    target_w = max(w_orig, 512)
                    current_frame_tensor = F.interpolate(current_frame_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

                if output_res_scale_factor != 1.0:
                    current_frame_tensor = F.interpolate(current_frame_tensor, size=(output_h, output_w), mode='bilinear', align_corners=False)
                I1_padded = self._pad_image_for_video(current_frame_tensor, model_scale_factor, use_fp16)

                if interpolation_exp > 0:
                    # Use standard recursive interpolation
                    interpolated_tensors = recursive_interpolate_video_frames(I0_padded, I1_padded, interpolation_exp, self.model, model_scale_factor)
                    
                    for mid_tensor in interpolated_tensors:
                        self._save_frame(mid_tensor, frames_output_dir / f"frame_{saved_frame_count:07d}.png", output_h, output_w)
                        saved_frame_count += 1
                
                self._save_frame(I1_padded, frames_output_dir / f"frame_{saved_frame_count:07d}.png", output_h, output_w)
                saved_frame_count += 1
                
                pbar.update(1)
                progress(frame_idx / (total_frames -1), desc=f"Processed frame {frame_idx}")


            pbar.close()
            status_updates.append(f"Frame generation complete. Total frames for output: {saved_frame_count}")
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-r', str(final_fps),
                '-i', frames_output_dir / 'frame_%07d.png',
                '-s', f'{output_w}x{output_h}',
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1',
                '-color_primaries', 'bt709',
                '-color_trc', 'bt709', 
                '-colorspace', 'bt709',
                '-movflags', '+faststart',
                final_output_video_path
            ]
            status_updates.append("Creating video from frames...")
            progress(1, desc="Assembling video")
            
            success_video, msg_video = run_ffmpeg_command(ffmpeg_cmd, operation_dir)
            if not success_video:
                raise Exception(f"FFmpeg error during video creation: {msg_video}")
            status_updates.append(f"Video created at {final_output_video_path}")

            status_updates.append("Attempting audio transfer...")
            progress(1, desc="Adding audio")
            audio_success, audio_msg = transfer_audio(input_path, final_output_video_path, operation_dir)
            status_updates.append(f"Audio transfer: {audio_msg}")
            
            return str(final_output_video_path), "\n".join(status_updates)

        except Exception as e:
            # Re-raise to be caught by Gradio
            raise e
        finally:
            if operation_dir and operation_dir.exists():
                shutil.rmtree(operation_dir)
            torch.set_default_dtype(original_torch_dtype) 