import shutil
import datetime
from pathlib import Path
from PIL import Image
import cv2
from enum import Enum
from typing import Union

from rife_app.config import DEVICE, CHAINED_TMP_DIR, VIDEO_TMP_DIR
from rife_app.utils.framing import get_video_info, extract_frames
from rife_app.utils.ffmpeg import run_ffmpeg_command
from rife_app.services.image_interpolator import ImageInterpolator
from rife_app.services.chained_simple import SimpleChainedInterpolator

class InterpolationMethod(Enum):
    """Defines available interpolation methods for video chaining."""
    IMAGE_INTERPOLATION = "image_interpolation"
    DISK_BASED = "disk_based"
    SIMPLE = "simple"  # New simple method

class ChainedInterpolator:
    """Enhanced video chaining with configurable interpolation methods."""
    
    def __init__(self, model):
        self.model = model
        self.image_interpolator = ImageInterpolator(model)
        self.simple_interpolator = SimpleChainedInterpolator(model)
        
    @classmethod
    def get_available_methods(cls):
        """Returns list of available interpolation methods."""
        return [method.value for method in InterpolationMethod]

    def _extract_boundary_frames(self, video_path, position='last'):
        """
        Frame-perfect boundary extraction using existing extract_frames function.
        
        Args:
            video_path: Path to video file
            position: 'first' or 'last' frame to extract
        
        Returns:
            PIL.Image: High-quality extracted frame
        """
        print(f"📸 Extracting {position} frame from {Path(video_path).name}")
        
        try:
            # Get video info to determine frame numbers
            info = get_video_info(Path(video_path))
            if not info:
                raise Exception(f"Could not read video info from {video_path}")
            
            frame_count = info['frame_count']
            if frame_count <= 0:
                raise Exception(f"Invalid frame count: {frame_count}")
            
            # Determine frame number based on position
            if position == 'last':
                frame_num = frame_count
            elif position == 'first':
                frame_num = 1
            else:
                raise Exception(f"Invalid position: {position}. Use 'first' or 'last'")
            
            # Extract the frame using existing function
            start_frame, end_frame = extract_frames(Path(video_path), frame_num, frame_num)
            
            # Return the appropriate frame
            frame_pil = start_frame if start_frame else end_frame
            
            if frame_pil is None:
                raise Exception(f"Frame extraction returned None for {position} frame")
            
            print(f"✅ Successfully extracted {position} frame: {frame_pil.size}")
            return frame_pil
            
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            raise
    
    def _create_transition_segment(self, start_frame_pil, end_frame_pil, num_passes, segment_path, target_fps, method=InterpolationMethod.IMAGE_INTERPOLATION):
        """
        Create transition video using specified interpolation method.
        
        Args:
            start_frame_pil: Starting frame (PIL Image)
            end_frame_pil: Ending frame (PIL Image) 
            num_passes: Number of interpolation passes
            segment_path: Output path for transition video
            target_fps: Target frame rate for video
            method: InterpolationMethod to use for transition creation
        
        Returns:
            str: Path to created transition video
        """
        print(f"🎬 Creating transition segment: {segment_path.name}")
        print(f"  - From frame: {start_frame_pil.size}")
        print(f"  - To frame: {end_frame_pil.size}")
        print(f"  - Target FPS: {target_fps}")
        print(f"  - Passes: {num_passes}")
        print(f"  - Method: {method.value}")
        
        try:
            if method == InterpolationMethod.IMAGE_INTERPOLATION:
                # Use ImageInterpolator service for proven interpolation logic
                video_path, status_msg = self.image_interpolator.interpolate(
                    img0_pil=start_frame_pil,
                    img1_pil=end_frame_pil,
                    num_passes=num_passes,
                    fps=target_fps,  # Use target FPS for consistency
                    use_disk_based=False  # Use memory-based for better quality
                )
                
            elif method == InterpolationMethod.DISK_BASED:
                # Use disk-based interpolation for memory efficiency
                video_path, status_msg = self.image_interpolator.interpolate(
                    img0_pil=start_frame_pil,
                    img1_pil=end_frame_pil,
                    num_passes=num_passes,
                    fps=target_fps,  # Use target FPS for consistency
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
        
        # Use simple method if selected
        if method == InterpolationMethod.SIMPLE:
            return self.simple_interpolator.interpolate(
                anchor_video_path, middle_video_path, end_video_path, 
                num_passes, final_fps
            )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        op_dir = CHAINED_TMP_DIR / f"chained_{timestamp}"
        op_dir.mkdir(parents=True)
        
        print(f"🎬 Starting enhanced chained interpolation")
        print(f"📁 Working directory: {op_dir}")
        print(f"🔗 Chain: {Path(anchor_video_path).name} -> {Path(middle_video_path).name} -> {Path(end_video_path).name}")
        print(f"⚙️ Interpolation method: {method.value}")
        print(f"🔢 Passes: {num_passes} (duration controlled by passes, not fixed duration)")
        
        try:
            # Simple validation
            print("📊 Validating input videos...")
            
            # Get individual video info
            anchor_info = get_video_info(Path(anchor_video_path))
            middle_info = get_video_info(Path(middle_video_path))
            end_info = get_video_info(Path(end_video_path))
            
            if not all([anchor_info, middle_info, end_info]):
                raise Exception("Could not read properties from one or more videos.")

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
                final_fps,
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
                final_fps,
                method
            )
            
            print("✅ All transition segments created successfully")

            # Phase 3: Simple FPS conversion if needed
            print(f"\n⚙️ Phase 3: Preparing videos for concatenation...")
            
            # Check if we need to convert FPS for any videos
            needs_fps_conversion = any([
                anchor_info['fps'] != final_fps,
                middle_info['fps'] != final_fps,
                end_info['fps'] != final_fps
            ])
            
            if needs_fps_conversion:
                print(f"📊 Converting videos to {final_fps} FPS for consistency")
                anchor_processed_path = segments_dir / "anchor_fps.mp4"
                middle_processed_path = segments_dir / "middle_fps.mp4"
                end_processed_path = segments_dir / "end_fps.mp4"
                
                for input_path, output_path, name, video_info in [
                    (anchor_video_path, anchor_processed_path, "anchor", anchor_info), 
                    (middle_video_path, middle_processed_path, "middle", middle_info), 
                    (end_video_path, end_processed_path, "end", end_info)
                ]:
                    if video_info['fps'] != final_fps:
                        print(f"🔄 Converting {name} video from {video_info['fps']} to {final_fps} FPS")
                        cmd = [
                            'ffmpeg', '-y', '-i', input_path, 
                            '-r', str(final_fps),
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                            '-pix_fmt', 'yuv420p',
                            '-an', output_path
                        ]
                        success, msg = run_ffmpeg_command(cmd)
                        if not success: 
                            raise Exception(f"FFmpeg error converting {name}: {msg}")
                    else:
                        # Just copy the file if FPS already matches
                        shutil.copy2(input_path, output_path)
                        print(f"✅ {name.capitalize()} video already at {final_fps} FPS")
            else:
                print("✅ All videos already at target FPS")
                # Use original paths
                anchor_processed_path = Path(anchor_video_path)
                middle_processed_path = Path(middle_video_path)
                end_processed_path = Path(end_video_path)

            # Phase 4: Simple validation
            print(f"\n🔍 Phase 4: Validating all segments...")
            
            all_segments = [
                ("Anchor video", anchor_processed_path),
                ("Transition 1", transition1_path),
                ("Middle video", middle_processed_path),
                ("Transition 2", transition2_path),
                ("End video", end_processed_path)
            ]
            
            for i, (name, path) in enumerate(all_segments, 1):
                if not path.exists():
                    raise Exception(f"Missing segment: {path}")
                
                file_size = path.stat().st_size
                if file_size == 0:
                    raise Exception(f"Empty segment: {path}")
                
                file_size_mb = file_size / (1024 * 1024)
                print(f"{i}. {name}: {file_size_mb:.1f} MB")

            # Phase 5: Simple concatenation
            print(f"\n🔗 Phase 5: Concatenating final video...")
            
            concat_list_path = op_dir / "concat_list.txt"
            with open(concat_list_path, 'w') as f:
                for name, path in all_segments:
                    f.write(f"file '{path.resolve()}'\n")
            
            final_video_path = VIDEO_TMP_DIR / f"chained_output_{timestamp}.mp4"
            
            # Simple concatenation
            cmd_concat = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                '-i', concat_list_path, 
                '-c', 'copy',
                '-an',  # Remove audio
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
            
            print(f"\n🎉 Chained interpolation completed successfully!")
            print(f"📁 Output: {final_video_path.name}")
            print(f"📊 Size: {final_size_mb:.1f} MB")
            print(f"🔗 Structure: [Video1] -> [Transition] -> [Video2] -> [Transition] -> [Video3]")
            
            status_message = f"Chained interpolation successful! Generated {final_size_mb:.1f} MB video."
            
            return str(final_video_path), status_message
        
        except Exception as e:
            print(f"❌ Chained interpolation failed: {e}")
            raise e
        finally:
            # Cleanup temporary directory
            if op_dir.exists():
                shutil.rmtree(op_dir)
                print(f"🧹 Cleaned up temporary directory: {op_dir}") 