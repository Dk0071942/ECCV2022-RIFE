import shutil
import datetime
from pathlib import Path
from PIL import Image
import cv2
from enum import Enum
from typing import Union

from rife_app.config import DEVICE, CHAINED_TMP_DIR, VIDEO_TMP_DIR
from rife_app.utils.framing import get_video_info, extract_frames
from rife_app.utils.ffmpeg import run_ffmpeg_command, scale_and_pad_image
from rife_app.services.image_interpolator import ImageInterpolator

class InterpolationMethod(Enum):
    """Defines available interpolation methods for video chaining."""
    IMAGE_INTERPOLATION = "image_interpolation"
    DISK_BASED = "disk_based"

class ChainedInterpolator:
    """Enhanced video chaining with configurable interpolation methods."""
    
    def __init__(self, model):
        self.model = model
        self.image_interpolator = ImageInterpolator(model)
        
    @classmethod
    def get_available_methods(cls):
        """Returns list of available interpolation methods."""
        return [method.value for method in InterpolationMethod]

    def _extract_boundary_frames(self, video_path, position='last'):
        """
        Enhanced frame extraction with validation.
        
        Args:
            video_path: Path to video file
            position: 'first' or 'last' frame to extract
        
        Returns:
            PIL.Image: Extracted frame
        """
        print(f"📸 Extracting {position} frame from {Path(video_path).name}")
        
        try:
            video_info = get_video_info(Path(video_path))
            if not video_info:
                raise Exception(f"Could not read video info from {video_path}")
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            frame_count = video_info['frame_count']
            if position == 'last':
                target_frame = frame_count - 1
            else:  # first
                target_frame = 0
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame_bgr = cap.read()
            cap.release()
            
            if not ret or frame_bgr is None:
                raise Exception(f"Could not read {position} frame from {video_path}")
            
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            print(f"✅ Successfully extracted {position} frame: {frame_pil.size}")
            return frame_pil
            
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            raise
    
    def _create_transition_segment(self, start_frame_pil, end_frame_pil, num_passes, segment_path, method=InterpolationMethod.IMAGE_INTERPOLATION):
        """
        Create transition video using specified interpolation method.
        
        Args:
            start_frame_pil: Starting frame (PIL Image)
            end_frame_pil: Ending frame (PIL Image) 
            num_passes: Number of interpolation passes
            segment_path: Output path for transition video
            method: InterpolationMethod to use for transition creation
        
        Returns:
            str: Path to created transition video
        """
        print(f"🎬 Creating transition segment: {segment_path.name}")
        print(f"  - From frame: {start_frame_pil.size}")
        print(f"  - To frame: {end_frame_pil.size}")
        print(f"  - Passes: {num_passes}")
        print(f"  - Method: {method.value}")
        
        try:
            if method == InterpolationMethod.IMAGE_INTERPOLATION:
                # Use ImageInterpolator service for proven interpolation logic
                video_path, status_msg = self.image_interpolator.interpolate(
                    img0_pil=start_frame_pil,
                    img1_pil=end_frame_pil,
                    num_passes=num_passes,
                    fps=25,  # Fixed FPS for consistency
                    use_disk_based=False  # Use memory-based for better quality
                )
                
            elif method == InterpolationMethod.DISK_BASED:
                # Use disk-based interpolation for memory efficiency
                video_path, status_msg = self.image_interpolator.interpolate(
                    img0_pil=start_frame_pil,
                    img1_pil=end_frame_pil,
                    num_passes=num_passes,
                    fps=25,
                    use_disk_based=True  # Use disk-based for memory efficiency
                )
                
            else:
                raise Exception(f"Unsupported interpolation method: {method}")
            
            if not video_path:
                raise Exception(f"Interpolation failed: {status_msg}")
            
            # Move the generated video to our target location
            import shutil
            shutil.move(video_path, segment_path)
            
            if not segment_path.exists():
                raise Exception(f"Transition segment not created: {segment_path}")
            
            segment_size = segment_path.stat().st_size
            print(f"✅ Transition segment created: {segment_size} bytes")
            print(f"   Method: {method.value}, Status: {status_msg}")
            
            return str(segment_path)
            
        except Exception as e:
            print(f"❌ Transition creation failed: {e}")
            raise

    def interpolate(self, anchor_video_path, middle_video_path, end_video_path, num_passes, final_fps, method: Union[str, InterpolationMethod] = InterpolationMethod.IMAGE_INTERPOLATION):
        """
        Enhanced chained interpolation with configurable methods.
        
        Creates smooth transitions between three videos by:
        1. Extracting boundary frames from each video
        2. Using specified interpolation method to create high-quality transitions  
        3. Concatenating: [video1] -> [transition1] -> [video2] -> [transition2] -> [video3]
        
        Args:
            anchor_video_path: Path to first video
            middle_video_path: Path to middle video
            end_video_path: Path to final video
            num_passes: Number of interpolation passes (affects transition duration)
            final_fps: Target FPS for final video
            method: Interpolation method to use (InterpolationMethod enum or string)
        
        Returns:
            tuple: (video_path, status_message)
        """
        if not all([anchor_video_path, middle_video_path, end_video_path]):
            raise Exception("Anchor video, middle video, and end video are all required.")
        if num_passes <= 0 or final_fps <= 0:
            raise Exception("Number of passes and FPS must be positive.")
        
        # Handle method parameter (string or enum)
        if isinstance(method, str):
            try:
                method = InterpolationMethod(method)
            except ValueError:
                raise Exception(f"Invalid interpolation method: {method}. Available: {self.get_available_methods()}")
        elif not isinstance(method, InterpolationMethod):
            raise Exception(f"Method must be InterpolationMethod enum or string. Got: {type(method)}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        op_dir = CHAINED_TMP_DIR / f"chained_{timestamp}"
        op_dir.mkdir(parents=True)
        
        print(f"🎬 Starting enhanced chained interpolation")
        print(f"📁 Working directory: {op_dir}")
        print(f"🔗 Chain: {Path(anchor_video_path).name} -> {Path(middle_video_path).name} -> {Path(end_video_path).name}")
        print(f"⚙️ Interpolation method: {method.value}")
        print(f"🔢 Passes: {num_passes} (duration controlled by passes, not fixed duration)")
        
        try:
            # Validate all input videos
            print("📊 Validating input videos...")
            anchor_info = get_video_info(Path(anchor_video_path))
            middle_info = get_video_info(Path(middle_video_path))
            end_info = get_video_info(Path(end_video_path))
            
            if not all([anchor_info, middle_info, end_info]):
                raise Exception("Could not read properties from one or more videos.")
            
            # Use middle video resolution as target
            target_w, target_h = middle_info['width'], middle_info['height']
            print(f"🎯 Target resolution: {target_w}x{target_h}")

            # Setup working directories
            segments_dir = op_dir / "segments"
            segments_dir.mkdir()

            # Phase 1: Extract boundary frames using enhanced extraction
            print("\n📸 Phase 1: Extracting boundary frames...")
            
            anchor_last_frame = self._extract_boundary_frames(anchor_video_path, 'last')
            middle_first_frame = self._extract_boundary_frames(middle_video_path, 'first')
            middle_last_frame = self._extract_boundary_frames(middle_video_path, 'last')
            end_first_frame = self._extract_boundary_frames(end_video_path, 'first')
            
            print("✅ All boundary frames extracted successfully")

            # Phase 2: Create transition segments using ImageInterpolator
            print(f"\n🎨 Phase 2: Creating transition segments with {num_passes} passes...")
            
            # Transition 1: anchor_last -> middle_first
            transition1_path = segments_dir / "transition1.mp4"
            print(f"Creating transition 1: {Path(anchor_video_path).name} -> {Path(middle_video_path).name}")
            self._create_transition_segment(
                anchor_last_frame, 
                middle_first_frame, 
                num_passes, 
                transition1_path,
                method
            )
            
            # Transition 2: middle_last -> end_first  
            transition2_path = segments_dir / "transition2.mp4"
            print(f"Creating transition 2: {Path(middle_video_path).name} -> {Path(end_video_path).name}")
            self._create_transition_segment(
                middle_last_frame, 
                end_first_frame, 
                num_passes, 
                transition2_path,
                method
            )
            
            print("✅ All transition segments created successfully")

            # Phase 3: Prepare main videos with consistent encoding
            print(f"\n⚙️ Phase 3: Re-encoding main videos for consistency...")
            
            anchor_reencoded_path = segments_dir / "anchor_reencoded.mp4"
            middle_reencoded_path = segments_dir / "middle_reencoded.mp4"
            end_reencoded_path = segments_dir / "end_reencoded.mp4"
            
            # Use scaling filter for consistent dimensions
            vf_filter = f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=1,pad=w={target_w}:h={target_h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black"
            
            for input_path, output_path, name in [
                (anchor_video_path, anchor_reencoded_path, "anchor"), 
                (middle_video_path, middle_reencoded_path, "middle"), 
                (end_video_path, end_reencoded_path, "end")
            ]:
                print(f"Re-encoding {name} video: {Path(input_path).name}")
                cmd = [
                    'ffmpeg', '-y', '-i', input_path, 
                    '-vf', vf_filter, 
                    '-r', str(final_fps), 
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', 
                    '-pix_fmt', 'yuv420p',
                    '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                    '-movflags', '+faststart', '-an', output_path
                ]
                success, msg = run_ffmpeg_command(cmd)
                if not success: 
                    raise Exception(f"FFmpeg error re-encoding {name}: {msg}")
                
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise Exception(f"Re-encoded {name} video failed: {output_path}")
                
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✅ {name.capitalize()} video re-encoded: {file_size_mb:.1f} MB")

            # Phase 4: Validate all segments before concatenation
            print(f"\n🔍 Phase 4: Validating all segments...")
            
            all_segments = [
                ("Anchor video", anchor_reencoded_path),
                ("Transition 1", transition1_path),
                ("Middle video", middle_reencoded_path),
                ("Transition 2", transition2_path),
                ("End video", end_reencoded_path)
            ]
            
            for i, (name, path) in enumerate(all_segments, 1):
                if not path.exists():
                    raise Exception(f"Missing segment: {path}")
                
                file_size = path.stat().st_size
                if file_size == 0:
                    raise Exception(f"Empty segment: {path}")
                
                # Get video info
                try:
                    video_info = get_video_info(path)
                    if video_info:
                        duration = video_info.get('duration', 0)
                        frame_count = video_info.get('frame_count', 0)
                        duration_str = f"{duration:.2f}s" if duration else "unknown"
                    else:
                        duration_str = "unknown"
                        frame_count = "unknown"
                except:
                    duration_str = "unknown"
                    frame_count = "unknown"
                
                file_size_mb = file_size / (1024 * 1024)
                print(f"{i}. {name}: {file_size_mb:.1f} MB, {duration_str}, {frame_count} frames")

            # Phase 5: Concatenate all segments
            print(f"\n🔗 Phase 5: Concatenating final video...")
            
            concat_list_path = op_dir / "concat_list.txt"
            with open(concat_list_path, 'w') as f:
                for name, path in all_segments:
                    f.write(f"file '{path.resolve()}'\n")
            
            final_video_path = VIDEO_TMP_DIR / f"chained_output_{timestamp}.mp4"
            
            cmd_concat = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                '-i', concat_list_path, 
                '-c', 'copy', '-r', str(final_fps), 
                final_video_path
            ]
            
            success, msg = run_ffmpeg_command(cmd_concat)
            if not success: 
                raise Exception(f"FFmpeg concatenation error: {msg}")
            
            # Final validation
            if not final_video_path.exists():
                raise Exception(f"Final video not created: {final_video_path}")
            if final_video_path.stat().st_size == 0:
                raise Exception(f"Final video is empty: {final_video_path}")
            
            final_size_mb = final_video_path.stat().st_size / (1024 * 1024)
            
            # Get final video info
            final_info = get_video_info(final_video_path)
            if final_info:
                total_duration = final_info.get('duration', 0)
                total_frames = final_info.get('frame_count', 0)
                duration_str = f"{total_duration:.2f}s" if total_duration else "unknown"
                print(f"📊 Final video stats: {final_size_mb:.1f} MB, {duration_str}, {total_frames} frames")
            
            print(f"\n🎉 Enhanced chained interpolation completed successfully!")
            print(f"📁 Output: {final_video_path.name}")
            print(f"🔗 Structure: [Video1] -> [Smooth Transition] -> [Video2] -> [Smooth Transition] -> [Video3]")
            print(f"✨ Enhanced with proven ImageInterpolator technology!")
            
            return str(final_video_path), f"Enhanced chained interpolation successful! Generated {final_size_mb:.1f} MB video with smooth image-to-image transitions."
        
        except Exception as e:
            print(f"❌ Chained interpolation failed: {e}")
            raise e
        finally:
            # Cleanup temporary directory
            if op_dir.exists():
                shutil.rmtree(op_dir)
                print(f"🧹 Cleaned up temporary directory: {op_dir}") 