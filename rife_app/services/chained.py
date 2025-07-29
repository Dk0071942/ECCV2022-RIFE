import shutil
import datetime
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from rife_app.config import CHAINED_TMP_DIR, VIDEO_TMP_DIR
from rife_app.services.image_interpolator import ImageInterpolator
from rife_app.utils.ffmpeg import run_ffmpeg_command


class ChainedInterpolator:
    """Simplified video chaining - extract all frames, interpolate boundaries, combine."""
    
    def __init__(self, model):
        self.model = model
        self.image_interpolator = ImageInterpolator(model)
    
    def _extract_all_frames(self, video_path, output_dir):
        """Extract all frames from a video to a directory using color-safe conversion."""
        print(f"📸 Extracting frames from {Path(video_path).name} with color-safe conversion")
        
        import subprocess
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Extract frames using FFmpeg with color-safe conversion (Video → PNG)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vf', 'scale=in_range=limited:out_range=full,format=rgb24',
            '-color_range', 'pc',
            str(output_dir / 'frame_%07d.png')
        ]
        
        print(f"🎨 Using color-safe extraction: limited range → full range")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ FFmpeg extraction failed, falling back to OpenCV")
            # Fallback to OpenCV extraction
            return self._extract_all_frames_opencv_fallback(video_path, output_dir)
        
        # Collect extracted frames
        frames = sorted(list(output_dir.glob("frame_*.png")))
        
        print(f"✅ Extracted {len(frames)} frames with proper color conversion")
        return frames, fps
    
    def _extract_all_frames_opencv_fallback(self, video_path, output_dir):
        """Fallback frame extraction using OpenCV (original method)."""
        print(f"📸 Extracting frames from {Path(video_path).name} (OpenCV fallback)")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        frame_idx = 0
        
        pbar = tqdm(total=frame_count, desc=f"Extracting frames from {Path(video_path).name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as PNG
            frame_path = output_dir / f"frame_{frame_idx:07d}.png"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame_path)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        print(f"✅ Extracted {len(frames)} frames")
        return frames, fps
    
    def _get_frame_as_pil(self, frame_path):
        """Load a frame from disk as PIL Image."""
        frame_bgr = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def interpolate(self, anchor_video_path, middle_video_path, end_video_path, num_passes, final_fps, method=None, use_disk_based=False):
        """
        Simple chained interpolation:
        1. Extract all frames from all videos
        2. Interpolate between boundary frames
        3. Combine all frames into output video
        
        Args:
            anchor_video_path: Path to first video
            middle_video_path: Path to second video  
            end_video_path: Path to third video
            num_passes: Number of interpolation passes (affects transition duration)
            final_fps: Target FPS for output video
            method: Ignored (kept for compatibility)
            use_disk_based: Whether to use disk-based interpolation
            
        Returns:
            tuple: (output_path, status_message)
        """
        if not all([anchor_video_path, middle_video_path, end_video_path]):
            raise Exception("All three videos are required.")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        op_dir = CHAINED_TMP_DIR / f"simple_chained_{timestamp}"
        op_dir.mkdir(parents=True)
        
        print(f"🎬 Starting simple chained interpolation")
        print(f"📁 Working directory: {op_dir}")
        print(f"🔗 Chain: {Path(anchor_video_path).name} -> {Path(middle_video_path).name} -> {Path(end_video_path).name}")
        print(f"🔢 Passes: {num_passes} | Target FPS: {final_fps}")
        
        try:
            # Create directories for frames
            anchor_frames_dir = op_dir / "anchor_frames"
            middle_frames_dir = op_dir / "middle_frames"
            end_frames_dir = op_dir / "end_frames"
            transition1_frames_dir = op_dir / "transition1_frames"
            transition2_frames_dir = op_dir / "transition2_frames"
            all_frames_dir = op_dir / "all_frames"
            
            for d in [anchor_frames_dir, middle_frames_dir, end_frames_dir, 
                     transition1_frames_dir, transition2_frames_dir, all_frames_dir]:
                d.mkdir()
            
            # Phase 1: Extract all frames from all videos
            print("\n📸 Phase 1: Extracting frames from videos...")
            anchor_frames, anchor_fps = self._extract_all_frames(anchor_video_path, anchor_frames_dir)
            middle_frames, middle_fps = self._extract_all_frames(middle_video_path, middle_frames_dir)
            end_frames, end_fps = self._extract_all_frames(end_video_path, end_frames_dir)
            
            if not anchor_frames or not middle_frames or not end_frames:
                raise Exception("No frames extracted from one or more videos")
            
            # Phase 2: Create transitions between videos
            print("\n🎨 Phase 2: Creating transitions between videos...")
            
            # Transition 1: anchor -> middle
            print(f"\nCreating transition 1: {Path(anchor_video_path).name} -> {Path(middle_video_path).name}")
            last_frame_anchor = self._get_frame_as_pil(anchor_frames[-1])
            first_frame_middle = self._get_frame_as_pil(middle_frames[0])
            
            transition1_video_path, status1 = self.image_interpolator.interpolate(
                img0_pil=last_frame_anchor,
                img1_pil=first_frame_middle,
                num_passes=num_passes,
                fps=final_fps,
                use_disk_based=use_disk_based
            )
            
            if not transition1_video_path:
                raise Exception(f"Transition 1 creation failed: {status1}")
            
            # Transition 2: middle -> end
            print(f"\nCreating transition 2: {Path(middle_video_path).name} -> {Path(end_video_path).name}")
            last_frame_middle = self._get_frame_as_pil(middle_frames[-1])
            first_frame_end = self._get_frame_as_pil(end_frames[0])
            
            transition2_video_path, status2 = self.image_interpolator.interpolate(
                img0_pil=last_frame_middle,
                img1_pil=first_frame_end,
                num_passes=num_passes,
                fps=final_fps,
                use_disk_based=use_disk_based
            )
            
            if not transition2_video_path:
                raise Exception(f"Transition 2 creation failed: {status2}")
            
            # Extract frames from transition videos
            print("\nExtracting transition frames...")
            
            # Extract transition 1 frames
            cap1 = cv2.VideoCapture(transition1_video_path)
            transition1_frames = []
            idx = 0
            while True:
                ret, frame = cap1.read()
                if not ret:
                    break
                frame_path = transition1_frames_dir / f"frame_{idx:05d}.png"
                cv2.imwrite(str(frame_path), frame)
                transition1_frames.append(frame_path)
                idx += 1
            cap1.release()
            
            # Extract transition 2 frames
            cap2 = cv2.VideoCapture(transition2_video_path)
            transition2_frames = []
            idx = 0
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                frame_path = transition2_frames_dir / f"frame_{idx:05d}.png"
                cv2.imwrite(str(frame_path), frame)
                transition2_frames.append(frame_path)
                idx += 1
            cap2.release()
            
            print(f"✅ Created {len(transition1_frames)} + {len(transition2_frames)} transition frames")
            
            # Phase 3: Combine all frames in order
            print("\n🔗 Phase 3: Combining all frames...")
            
            frame_counter = 0
            
            # Copy anchor frames (excluding last frame)
            print(f"Copying {len(anchor_frames)-1} frames from anchor video...")
            for frame_path in anchor_frames[:-1]:
                dest_path = all_frames_dir / f"frame_{frame_counter:07d}.png"
                shutil.copy2(frame_path, dest_path)
                frame_counter += 1
            
            # Copy transition 1 frames
            print(f"Copying {len(transition1_frames)} transition 1 frames...")
            for frame_path in transition1_frames:
                dest_path = all_frames_dir / f"frame_{frame_counter:07d}.png"
                shutil.copy2(frame_path, dest_path)
                frame_counter += 1
            
            # Copy middle frames (excluding first and last)
            print(f"Copying {len(middle_frames)-2} frames from middle video...")
            for frame_path in middle_frames[1:-1]:
                dest_path = all_frames_dir / f"frame_{frame_counter:07d}.png"
                shutil.copy2(frame_path, dest_path)
                frame_counter += 1
            
            # Copy transition 2 frames
            print(f"Copying {len(transition2_frames)} transition 2 frames...")
            for frame_path in transition2_frames:
                dest_path = all_frames_dir / f"frame_{frame_counter:07d}.png"
                shutil.copy2(frame_path, dest_path)
                frame_counter += 1
            
            # Copy end frames (excluding first frame)
            print(f"Copying {len(end_frames)-1} frames from end video...")
            for frame_path in end_frames[1:]:
                dest_path = all_frames_dir / f"frame_{frame_counter:07d}.png"
                shutil.copy2(frame_path, dest_path)
                frame_counter += 1
            
            print(f"✅ Total frames combined: {frame_counter}")
            
            # Phase 4: Create final video from all frames
            print("\n🎬 Phase 4: Creating final video...")
            
            final_video_path = VIDEO_TMP_DIR / f"simple_chained_output_{timestamp}.mp4"
            
            # Get dimensions from first frame
            first_frame = cv2.imread(str(all_frames_dir / "frame_0000000.png"))
            if first_frame is None:
                raise Exception("Could not read first frame")
            height, width = first_frame.shape[:2]
            
            # Create video using ffmpeg with color-safe conversion (PNG → Video)
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-color_range', '2',  # Specify input is full range (2 = pc/full)
                '-r', str(final_fps),
                '-i', str(all_frames_dir / 'frame_%07d.png'),
                '-vf', 'scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=limited,format=yuv420p',
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-color_range', 'tv',
                '-color_primaries', 'bt709',
                '-color_trc', 'bt709',
                '-colorspace', 'bt709',
                '-movflags', '+faststart',
                str(final_video_path)
            ]
            
            success, msg = run_ffmpeg_command(ffmpeg_cmd)
            if not success:
                raise Exception(f"FFmpeg error: {msg}")
            
            # Verify output
            if not final_video_path.exists() or final_video_path.stat().st_size == 0:
                raise Exception("Final video creation failed")
            
            final_size_mb = final_video_path.stat().st_size / (1024 * 1024)
            
            print(f"\n🎉 Simple chained interpolation completed successfully!")
            print(f"📁 Output: {final_video_path.name}")
            print(f"📊 Size: {final_size_mb:.1f} MB")
            print(f"🎞️ Total frames: {frame_counter}")
            print(f"⏱️ Duration: ~{frame_counter/final_fps:.1f} seconds at {final_fps} FPS")
            
            # Build detailed status message
            frame_breakdown = [
                f"{len(anchor_frames)-1} (video 1)",
                f"{len(transition1_frames)} (transition 1)",
                f"{len(middle_frames)-2} (video 2)",
                f"{len(transition2_frames)} (transition 2)",
                f"{len(end_frames)-1} (video 3)"
            ]
            
            status_message = (
                f"Success! Combined {' + '.join(frame_breakdown)} = {frame_counter} frames "
                f"into {final_size_mb:.1f} MB video"
            )
            
            return str(final_video_path), status_message
            
        except Exception as e:
            print(f"❌ Simple chained interpolation failed: {e}")
            raise e
        finally:
            # Cleanup temporary directory
            if op_dir.exists():
                shutil.rmtree(op_dir)
                print(f"🧹 Cleaned up temporary directory")