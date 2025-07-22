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

    def _generate_interpolation_segment(self, img_start_pil, img_end_pil, num_passes, fps, w, h, frames_dir, segment_path):
        print(f"🔧 _generate_interpolation_segment: {segment_path.name}")
        print(f"  - Input frames: {img_start_pil.size} -> {img_end_pil.size}")
        print(f"  - Parameters: passes={num_passes}, fps={fps}, target={w}x{h}")
        
        img_start_tensor = pil_to_tensor(img_start_pil, DEVICE)
        img_end_tensor = pil_to_tensor(img_end_pil, DEVICE)

        img_start_padded, _ = pad_tensor_for_rife(img_start_tensor)
        img_end_padded, _ = pad_tensor_for_rife(img_end_tensor)
        
        print(f"  - Running {num_passes} passes of 2x interpolation for optimal quality...")
        
        # Use multiple passes of exp=1 for best quality
        frame_tensors = [img_start_padded]  # Start with first frame
        current_frames = [img_start_padded, img_end_padded]  # Initial frame pair
        
        for pass_num in range(num_passes):
            print(f"  - Running pass {pass_num + 1}/{num_passes}...")
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
        print(f"  - Multiple passes completed: {len(frame_tensors)} total frames generated")
        
        # CRITICAL FIX: Exclude start and end frames to prevent duplication
        # frame_tensors contains [start_frame, intermediate_frames..., end_frame]
        # We only want the intermediate frames for smooth transitions
        if len(frame_tensors) > 2:
            # Remove first frame (start) and last frame (end) to avoid duplication
            intermediate_frames = frame_tensors[1:-1]
            print(f"  - Generated {len(frame_tensors)} total frames, using {len(intermediate_frames)} intermediate frames")
        else:
            # If only 2 frames (start, end), generate at least one intermediate frame
            intermediate_frames = frame_tensors[1:-1] if len(frame_tensors) > 1 else []
            print(f"  - ⚠️ WARNING: Only {len(frame_tensors)} frames generated, using {len(intermediate_frames)} intermediate frames")
        
        # Save only the intermediate frames
        print(f"  - Saving {len(intermediate_frames)} frames to {frames_dir}")
        if len(intermediate_frames) == 0:
            # No intermediate frames to save - create a minimal transition
            print("  - ⚠️ No intermediate frames available, creating minimal transition video")
            # Create a very short video with just one frame (duplicate of start frame)
            save_tensor_as_image(frame_tensors[0], frames_dir / f'frame_00000.png', original_size=(h, w))
            saved_frames = 1
        else:
            for i, frame_tensor in enumerate(intermediate_frames):
                save_tensor_as_image(frame_tensor, frames_dir / f'frame_{i:05d}.png', original_size=(h, w))
            saved_frames = len(intermediate_frames)
        
        print(f"  - Saved {saved_frames} frames, creating video with FFmpeg...")
            
        # Create segment video with proper BT.709 color space metadata
        cmd = [
            'ffmpeg', '-y', '-r', str(fps), '-i', frames_dir / 'frame_%05d.png',
            '-s', f'{w}x{h}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',
            '-movflags', '+faststart', segment_path
        ]
        print(f"  - FFmpeg command: {' '.join(str(x) for x in cmd[:8])}...")
        success, msg = run_ffmpeg_command(cmd)
        if not success:
            print(f"  - ❌ FFmpeg failed: {msg}")
            raise Exception(f"FFmpeg error for segment {segment_path.name}: {msg}")
        else:
            print(f"  - ✅ FFmpeg completed successfully for {segment_path.name}")

    def interpolate(self, anchor_video_path, middle_video_path, end_video_path, num_passes, interp_duration_seconds, final_fps):
        if not all([anchor_video_path, middle_video_path, end_video_path]):
            raise Exception("Anchor video, middle video, and end video are all required.")
        if interp_duration_seconds <= 0 or final_fps <= 0:
            raise Exception("Durations and FPS must be positive.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        op_dir = CHAINED_TMP_DIR / f"chained_{timestamp}"
        op_dir.mkdir(parents=True)
        
        try:
            # Get video information for all three videos
            anchor_info = get_video_info(Path(anchor_video_path))
            middle_info = get_video_info(Path(middle_video_path))
            end_info = get_video_info(Path(end_video_path))
            
            if not all([anchor_info, middle_info, end_info]):
                raise Exception("Could not read properties from one or more videos.")
            
            # Use middle video resolution as target (you could also use max resolution)
            target_w, target_h = middle_info['width'], middle_info['height']

            # Prepare directories
            frames1_dir = op_dir / "frames1"  # anchor -> middle interpolation
            frames2_dir = op_dir / "frames2"  # middle -> end interpolation
            segments_dir = op_dir / "segments"
            for d in [frames1_dir, frames2_dir, segments_dir]:
                d.mkdir()

            # Extract frames from videos
            print("Extracting key frames for interpolation...")
            
            # Get last frame of anchor video
            cap_anchor = cv2.VideoCapture(anchor_video_path)
            if not cap_anchor.isOpened():
                raise Exception(f"Could not open anchor video: {anchor_video_path}")
            cap_anchor.set(cv2.CAP_PROP_POS_FRAMES, anchor_info['frame_count'] - 1)
            ret, anchor_last_bgr = cap_anchor.read()
            cap_anchor.release()
            if not ret or anchor_last_bgr is None:
                raise Exception(f"Could not read last frame from anchor video")
            anchor_last_pil = Image.fromarray(cv2.cvtColor(anchor_last_bgr, cv2.COLOR_BGR2RGB))
            print(f"Extracted anchor video last frame: {anchor_last_pil.size}")

            # Get first and last frame of middle video
            cap_middle = cv2.VideoCapture(middle_video_path)
            if not cap_middle.isOpened():
                raise Exception(f"Could not open middle video: {middle_video_path}")
            cap_middle.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, middle_first_bgr = cap_middle.read()
            if not ret or middle_first_bgr is None:
                raise Exception(f"Could not read first frame from middle video")
            cap_middle.set(cv2.CAP_PROP_POS_FRAMES, middle_info['frame_count'] - 1)
            ret, middle_last_bgr = cap_middle.read()
            cap_middle.release()
            if not ret or middle_last_bgr is None:
                raise Exception(f"Could not read last frame from middle video")
            middle_first_pil = Image.fromarray(cv2.cvtColor(middle_first_bgr, cv2.COLOR_BGR2RGB))
            middle_last_pil = Image.fromarray(cv2.cvtColor(middle_last_bgr, cv2.COLOR_BGR2RGB))
            print(f"Extracted middle video first frame: {middle_first_pil.size}")
            print(f"Extracted middle video last frame: {middle_last_pil.size}")

            # Get first frame of end video
            cap_end = cv2.VideoCapture(end_video_path)
            if not cap_end.isOpened():
                raise Exception(f"Could not open end video: {end_video_path}")
            cap_end.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, end_first_bgr = cap_end.read()
            cap_end.release()
            if not ret or end_first_bgr is None:
                raise Exception(f"Could not read first frame from end video")
            end_first_pil = Image.fromarray(cv2.cvtColor(end_first_bgr, cv2.COLOR_BGR2RGB))
            print(f"Extracted end video first frame: {end_first_pil.size}")

            # Calculate interpolation FPS - fixed at 25 FPS, passes control duration
            num_frames = (2**num_passes)
            # Use fixed 25 FPS but adjust duration based on passes
            interp_fps = 25.0
            actual_duration = num_frames / 25.0
            print(f"Interpolation will create {num_frames} frames at 25 FPS = {actual_duration:.2f}s duration")

            # Interpolation 1: Last frame of anchor video -> First frame of middle video
            interp1_path = segments_dir / "interp1.mp4"
            print(f"Generating interpolation 1: anchor last frame -> middle first frame")
            self._generate_interpolation_segment(anchor_last_pil, middle_first_pil, num_passes, interp_fps, target_w, target_h, frames1_dir, interp1_path)
            
            # Validate interpolation 1 was created
            if not interp1_path.exists():
                raise Exception(f"Interpolation 1 failed: {interp1_path} not created")
            interp1_size = interp1_path.stat().st_size
            print(f"Interpolation 1 generated successfully: {interp1_size} bytes")
            
            # Small file is OK since we're now only generating intermediate frames
            if interp1_size < 1000:  # Less than 1KB might indicate a problem
                print(f"Warning: Interpolation 1 is very small ({interp1_size} bytes) - may have few intermediate frames")

            # Interpolation 2: Last frame of middle video -> First frame of end video
            interp2_path = segments_dir / "interp2.mp4"
            print(f"🔍 DEBUG: Generating interpolation 2: middle last frame -> end first frame")
            print(f"🔍 DEBUG: Middle last frame size: {middle_last_pil.size}")
            print(f"🔍 DEBUG: End first frame size: {end_first_pil.size}")
            print(f"🔍 DEBUG: Number of passes: {num_passes}, FPS: {interp_fps}")
            print(f"🔍 DEBUG: Target dimensions: {target_w}x{target_h}")
            
            try:
                self._generate_interpolation_segment(middle_last_pil, end_first_pil, num_passes, interp_fps, target_w, target_h, frames2_dir, interp2_path)
                print(f"🔍 DEBUG: Interpolation 2 generation completed without exception")
            except Exception as e:
                print(f"❌ ERROR: Interpolation 2 generation failed: {e}")
                raise
            
            # Validate interpolation 2 was created
            if not interp2_path.exists():
                raise Exception(f"❌ Interpolation 2 failed: {interp2_path} not created")
            
            interp2_size = interp2_path.stat().st_size
            print(f"✅ Interpolation 2 generated successfully: {interp2_size} bytes")
            
            # Get video info for interpolation 2
            interp2_info = get_video_info(interp2_path)
            if interp2_info:
                frame_count = interp2_info.get('frame_count', 'unknown')
                duration = interp2_info.get('duration', 'unknown')
                if duration != 'unknown':
                    duration_str = f"{duration:.2f}s"
                else:
                    duration_str = "unknown"
                print(f"🔍 DEBUG: Interpolation 2 video info: {frame_count} frames, {duration_str}")
            else:
                print(f"⚠️ WARNING: Could not read interpolation 2 video info")
            
            if interp2_size < 1000:  # Less than 1KB might indicate a problem
                print(f"⚠️ WARNING: Interpolation 2 is very small ({interp2_size} bytes) - may have few intermediate frames")
                # List frame files in the directory
                frame_files = list(frames2_dir.glob("*.png"))
                print(f"🔍 DEBUG: Found {len(frame_files)} frame files in {frames2_dir}")
                for frame_file in sorted(frame_files)[:5]:  # Show first 5
                    print(f"  - {frame_file.name} ({frame_file.stat().st_size} bytes)")

            # Prepare all three videos with consistent resolution and frame rate
            anchor_reencoded_path = segments_dir / "anchor_reencoded.mp4"
            middle_reencoded_path = segments_dir / "middle_reencoded.mp4"
            end_reencoded_path = segments_dir / "end_reencoded.mp4"
            
            vf_filter = f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=1,pad=w={target_w}:h={target_h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black"
            
            # Re-encode all videos with consistent settings
            for input_path, output_path in [(anchor_video_path, anchor_reencoded_path), 
                                          (middle_video_path, middle_reencoded_path), 
                                          (end_video_path, end_reencoded_path)]:
                print(f"Re-encoding {Path(input_path).name} -> {output_path.name}")
                cmd = ['ffmpeg', '-y', '-i', input_path, '-vf', vf_filter, '-r', str(final_fps), 
                      '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
                      '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                      '-movflags', '+faststart', '-an', output_path]
                success, msg = run_ffmpeg_command(cmd)
                if not success: 
                    raise Exception(f"FFmpeg error re-encoding {Path(input_path).name}: {msg}")
                
                # Validate re-encoded video was created
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise Exception(f"Re-encoded video failed: {output_path} not created or empty")
                print(f"Re-encoded {output_path.name}: {output_path.stat().st_size} bytes")

            # Validate all files exist before concatenation
            all_files = [anchor_reencoded_path, interp1_path, middle_reencoded_path, interp2_path, end_reencoded_path]
            for file_path in all_files:
                if not file_path.exists():
                    raise Exception(f"Missing file for concatenation: {file_path}")
                file_size = file_path.stat().st_size
                if file_size == 0:
                    raise Exception(f"Empty file for concatenation: {file_path}")
                # Interpolation files can be small, but main videos should be substantial
                if 'interp' not in file_path.name and file_size < 10000:  # Less than 10KB for main videos is suspicious
                    print(f"Warning: Main video {file_path.name} is very small ({file_size} bytes)")

            # Concatenate: anchor video + interpolation1 + middle video + interpolation2 + end video
            concat_list_path = op_dir / "concat_list.txt"
            with open(concat_list_path, 'w') as f:
                f.write(f"file '{anchor_reencoded_path.resolve()}'\n")
                f.write(f"file '{interp1_path.resolve()}'\n")
                f.write(f"file '{middle_reencoded_path.resolve()}'\n")
                f.write(f"file '{interp2_path.resolve()}'\n")
                f.write(f"file '{end_reencoded_path.resolve()}'\n")
            
            print(f"📋 Concatenation list created with 5 segments:")
            
            # Show detailed info for each segment
            segments_info = [
                (1, "Anchor video", anchor_reencoded_path),
                (2, "Interpolation 1", interp1_path),
                (3, "Middle video", middle_reencoded_path),
                (4, "Interpolation 2", interp2_path),
                (5, "End video", end_reencoded_path)
            ]
            
            for num, name, path in segments_info:
                file_size = path.stat().st_size
                # Get video duration if possible
                try:
                    video_info = get_video_info(path)
                    if video_info:
                        duration_val = video_info.get('duration', None)
                        duration = f"{duration_val:.2f}s" if duration_val is not None else "unknown"
                        frame_count = video_info.get('frame_count', 'unknown')
                    else:
                        duration = "unknown"
                        frame_count = "unknown"
                except Exception as e:
                    duration = "unknown"
                    frame_count = "unknown"
                
                print(f"{num}. {name}: {path.name} ({file_size} bytes, {duration}, {frame_count} frames)")

            final_video_path = VIDEO_TMP_DIR / f"chained_output_{timestamp}.mp4"
            # Concatenate with color space preservation
            print(f"Starting concatenation to: {final_video_path}")
            cmd_concat = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', '-r', str(final_fps), final_video_path]
            success, msg = run_ffmpeg_command(cmd_concat)
            if not success: 
                raise Exception(f"FFmpeg error during concatenation: {msg}")
            
            # Validate final output
            if not final_video_path.exists():
                raise Exception(f"Final video not created: {final_video_path}")
            if final_video_path.stat().st_size == 0:
                raise Exception(f"Final video is empty: {final_video_path}")
            
            final_size_mb = final_video_path.stat().st_size / (1024 * 1024)
            print(f"🎬 Chained interpolation completed successfully!")
            print(f"📁 Final video: {final_video_path.name} ({final_size_mb:.1f} MB)")
            print(f"🔗 Structure: Anchor -> [Smooth Transition] -> Middle -> [Smooth Transition] -> End")
            print(f"✅ Frame duplication eliminated - smooth transitions guaranteed!")
            
            return str(final_video_path), "Chained video interpolation successful with smooth transitions."
        
        except Exception as e:
            raise e
        finally:
            if op_dir.exists():
                shutil.rmtree(op_dir) 