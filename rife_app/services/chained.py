import shutil
import datetime
import time
import subprocess
from pathlib import Path
from PIL import Image
from enum import Enum
from typing import Union, Optional, Tuple

from rife_app.config import DEVICE, CHAINED_TMP_DIR, VIDEO_TMP_DIR
from rife_app.utils.framing import extract_frames, get_video_info, validate_temporal_alignment
from rife_app.utils.ffmpeg import run_ffmpeg_command
from rife_app.services.image_interpolator import ImageInterpolator

class InterpolationMethod(Enum):
    """Defines available interpolation methods for video chaining."""
    IMAGE_INTERPOLATION = "image_interpolation"
    DISK_BASED = "disk_based"

# Constants for robust operation
DEFAULT_FFMPEG_TIMEOUT = 180  # 3 minutes for most ffmpeg operations
DEFAULT_CLEANUP_RETRIES = 3
DEFAULT_CLEANUP_DELAY = 1.0

class ChainedInterpolator:
    """Chained video interpolation - combines Tab 1 (frame extraction) and Tab 2 (interpolation)."""
    
    def __init__(self, model):
        self.model = model
        self.image_interpolator = ImageInterpolator(model)
    
    def _safe_rmtree(self, path: Path, max_retries: int = DEFAULT_CLEANUP_RETRIES) -> bool:
        """Safely remove a directory tree with retries for Windows file locking issues."""
        if not path.exists():
            return True
            
        delay = DEFAULT_CLEANUP_DELAY
        for attempt in range(max_retries):
            try:
                shutil.rmtree(path)
                return True
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Permission error removing {path} (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed to remove {path} after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                print(f"❌ Unexpected error removing {path}: {e}")
                return False
        return False
    
    def _validate_video_compatibility(self, video_paths: list, target_fps: float) -> Tuple[bool, str]:
        """Validate that videos are compatible for chaining."""
        try:
            alignment_info = validate_temporal_alignment(video_paths, target_fps)
            
            if not alignment_info['videos']:
                return False, "Could not read video properties"
            
            # Check for critical incompatibilities
            if not alignment_info['resolution_consistent']:
                resolutions = [(v['resolution']) for v in alignment_info['videos']]
                return False, f"Resolution mismatch detected: {resolutions}. All videos must have the same resolution."
            
            # Report FPS consistency (warning, not error)
            if not alignment_info['fps_consistent']:
                fps_values = [v['fps'] for v in alignment_info['videos']]
                print(f"⚠️ FPS inconsistency detected: {fps_values}. Videos will be standardized to {target_fps} FPS.")
            
            return True, "Videos are compatible"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _get_video_dimensions(self, video_path: Path) -> Optional[Tuple[int, int]]:
        """Get video dimensions using ffprobe with timeout."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and 'x' in result.stdout:
                width, height = map(int, result.stdout.strip().split('x'))
                return (width, height)
        except Exception as e:
            print(f"⚠️ Could not get dimensions for {video_path.name}: {e}")
        return None
        
    @classmethod
    def get_available_methods(cls):
        """Returns list of available interpolation methods."""
        return [method.value for method in InterpolationMethod]
    
    def interpolate(self, anchor_video_path, middle_video_path, end_video_path, num_passes, final_fps, method: Union[str, InterpolationMethod] = InterpolationMethod.IMAGE_INTERPOLATION):
        """
        Chained interpolation - exactly like Tab 1 + Tab 2:
        1. Extract last frame from video 1 and first frame from video 2 (Tab 1)
        2. Create interpolation between them (Tab 2)
        3. Extract last frame from video 2 and first frame from video 3 (Tab 1)
        4. Create interpolation between them (Tab 2)
        5. Concatenate all videos and transitions
        """
        # Enhanced input validation
        if not all([anchor_video_path, middle_video_path, end_video_path]):
            raise Exception("Anchor video, middle video, and end video are all required.")
        if num_passes <= 0 or final_fps <= 0:
            raise Exception("Number of passes and FPS must be positive.")
        
        # Validate files exist
        video_paths = [Path(p) for p in [anchor_video_path, middle_video_path, end_video_path]]
        for i, path in enumerate(video_paths, 1):
            if not path.exists():
                raise Exception(f"Video {i} not found: {path.name}")
            if path.stat().st_size == 0:
                raise Exception(f"Video {i} is empty: {path.name}")
        
        # Handle method parameter
        if isinstance(method, str):
            try:
                method = InterpolationMethod(method)
            except ValueError:
                raise Exception(f"Invalid interpolation method: {method}. Available: {self.get_available_methods()}")
        
        use_disk_based = (method == InterpolationMethod.DISK_BASED)
        
        # Create unique work directory with enhanced naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        work_dir = CHAINED_TMP_DIR / f"chained_{timestamp}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate video compatibility early
        print("🔍 Validating video compatibility...")
        compatible, validation_msg = self._validate_video_compatibility(
            [str(p) for p in video_paths], final_fps
        )
        if not compatible:
            if work_dir.exists():
                self._safe_rmtree(work_dir)
            raise Exception(f"Video compatibility check failed: {validation_msg}")
        print(f"✅ {validation_msg}")
        
        try:
            print(f"🎬 Starting chained interpolation")
            print(f"📁 Working directory: {work_dir}")
            print(f"🔗 Chain: {Path(anchor_video_path).name} -> {Path(middle_video_path).name} -> {Path(end_video_path).name}")
            print(f"⚙️ Method: {method.value}")
            
            # Step 1: Extract boundary frames (exactly like Tab 1)
            print("\n📸 Step 1: Extracting boundary frames...")
            
            # Get frame count for each video with enhanced error handling
            video_infos = []
            for i, video_path in enumerate(video_paths, 1):
                info = get_video_info(video_path)
                if not info:
                    raise Exception(f"Could not read properties for video {i}: {video_path.name}")
                if info['frame_count'] <= 0:
                    raise Exception(f"Video {i} has no frames: {video_path.name}")
                video_infos.append(info)
                print(f"  Video {i}: {info['frame_count']} frames, {info['fps']:.2f} FPS, {info['width']}x{info['height']}")
            
            anchor_info, middle_info, end_info = video_infos
            
            # Extract boundary frames with enhanced validation
            boundary_frames = []
            frame_descriptions = [
                (video_paths[0], anchor_info['frame_count'], anchor_info['frame_count'], "last frame of video 1"),
                (video_paths[1], 1, 1, "first frame of video 2"),
                (video_paths[1], middle_info['frame_count'], middle_info['frame_count'], "last frame of video 2"),
                (video_paths[2], 1, 1, "first frame of video 3")
            ]
            
            for video_path, start_frame, end_frame, description in frame_descriptions:
                print(f"  Extracting {description} (frame {start_frame})")
                try:
                    first_frame, last_frame = extract_frames(video_path, start_frame, end_frame)
                    extracted_frame = last_frame if start_frame == end_frame else first_frame
                    
                    if not extracted_frame:
                        raise Exception(f"Failed to extract {description} from {video_path.name}")
                    
                    # Validate frame dimensions
                    if not hasattr(extracted_frame, 'size') or extracted_frame.size[0] == 0 or extracted_frame.size[1] == 0:
                        raise Exception(f"Invalid frame extracted for {description}")
                    
                    boundary_frames.append(extracted_frame)
                    print(f"    ✅ Extracted {description}: {extracted_frame.size[0]}x{extracted_frame.size[1]}")
                    
                except Exception as e:
                    raise Exception(f"Frame extraction failed for {description}: {str(e)}")
            
            anchor_last, middle_first, middle_last, end_first = boundary_frames
            
            print("✅ All boundary frames extracted")
            
            # Step 2: Create interpolations (exactly like Tab 2)
            print(f"\n🎨 Step 2: Creating interpolated transitions ({num_passes} passes)...")
            
            # Create interpolations with enhanced error handling and validation
            transitions = []
            transition_pairs = [
                (anchor_last, middle_first, "transition 1 (video1→video2)"),
                (middle_last, end_first, "transition 2 (video2→video3)")
            ]
            
            for i, (frame_a, frame_b, description) in enumerate(transition_pairs, 1):
                print(f"  Creating {description}...")
                
                # Validate frame compatibility
                if frame_a.size != frame_b.size:
                    raise Exception(f"Frame size mismatch in {description}: {frame_a.size} vs {frame_b.size}")
                
                try:
                    transition_path, status = self.image_interpolator.interpolate(
                        frame_a, frame_b, num_passes, final_fps, use_disk_based
                    )
                    
                    if not transition_path:
                        raise Exception(f"Interpolation failed: {status}")
                    
                    # Validate output file
                    transition_file = Path(transition_path)
                    if not transition_file.exists():
                        raise Exception(f"Transition file not created: {transition_path}")
                    
                    if transition_file.stat().st_size == 0:
                        raise Exception(f"Transition file is empty: {transition_path}")
                    
                    transitions.append(transition_path)
                    print(f"    ✅ {description} created successfully ({transition_file.stat().st_size / (1024*1024):.1f} MB)")
                    print(f"    📊 Status: {status}")
                    
                except Exception as e:
                    raise Exception(f"Failed to create {description}: {str(e)}")
            
            transition1_path, transition2_path = transitions
            
            print("✅ All transitions created")
            
            # Step 3: Convert videos to target FPS if needed
            print(f"\n📊 Step 3: Preparing videos for concatenation...")
            
            segments_dir = work_dir / "segments"
            segments_dir.mkdir()
            
            # Convert each video to target FPS with enhanced processing
            processed_video_paths = []
            for i, (video_path, info) in enumerate(zip(video_paths, video_infos), 1):
                output_path = segments_dir / f"video{i}_converted.mp4"
                
                try:
                    if abs(info['fps'] - final_fps) > 0.1:  # Allow small tolerance
                        print(f"  Converting video {i} from {info['fps']:.2f} to {final_fps} FPS")
                        cmd = [
                            'ffmpeg', '-y', '-i', str(video_path),
                            '-r', str(final_fps),
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                            '-pix_fmt', 'yuv420p',
                            '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:fast=1',
                            '-color_primaries', 'bt709',
                            '-color_trc', 'bt709',
                            '-colorspace', 'bt709',
                            '-an',  # Remove audio
                            str(output_path)
                        ]
                        
                        # Use subprocess with timeout for better control
                        try:
                            result = subprocess.run(cmd, capture_output=True, text=True, 
                                                  timeout=DEFAULT_FFMPEG_TIMEOUT, check=True)
                            print(f"    ✅ Video {i} converted successfully")
                        except subprocess.TimeoutExpired:
                            raise Exception(f"Video {i} conversion timed out after {DEFAULT_FFMPEG_TIMEOUT}s")
                        except subprocess.CalledProcessError as e:
                            raise Exception(f"Video {i} conversion failed: {e.stderr}")
                    else:
                        print(f"  Video {i} already at target FPS ({info['fps']:.2f})")
                        shutil.copy2(video_path, output_path)
                    
                    # Validate converted file
                    if not output_path.exists() or output_path.stat().st_size == 0:
                        raise Exception(f"Video {i} conversion produced invalid output")
                    
                    processed_video_paths.append(output_path)
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    print(f"    📊 Video {i} processed: {file_size_mb:.1f} MB")
                    
                except Exception as e:
                    raise Exception(f"Failed to process video {i} ({video_path.name}): {str(e)}")
            
            # Copy transition videos with validation
            transition_copies = []
            for i, transition_path in enumerate([transition1_path, transition2_path], 1):
                transition_copy = segments_dir / f"transition{i}.mp4"
                try:
                    shutil.copy2(transition_path, transition_copy)
                    if not transition_copy.exists() or transition_copy.stat().st_size == 0:
                        raise Exception(f"Failed to copy transition {i}")
                    transition_copies.append(transition_copy)
                    file_size_mb = transition_copy.stat().st_size / (1024 * 1024)
                    print(f"    ✅ Transition {i} copied: {file_size_mb:.1f} MB")
                except Exception as e:
                    raise Exception(f"Failed to copy transition {i}: {str(e)}")
            
            transition1_copy, transition2_copy = transition_copies
            
            # Step 4: Concatenate all segments
            print("\n🔗 Step 4: Concatenating final video...")
            
            # Create concat list with enhanced validation
            concat_list_path = work_dir / "concat_list.txt"
            concat_segments = [
                (processed_video_paths[0], "Video 1"),
                (transition1_copy, "Transition 1"),
                (processed_video_paths[1], "Video 2"), 
                (transition2_copy, "Transition 2"),
                (processed_video_paths[2], "Video 3")
            ]
            
            print("  Preparing concatenation list...")
            try:
                with open(concat_list_path, 'w', encoding='utf-8') as f:
                    for segment_path, description in concat_segments:
                        # Validate segment before adding to list
                        if not segment_path.exists():
                            raise Exception(f"{description} file not found: {segment_path}")
                        if segment_path.stat().st_size == 0:
                            raise Exception(f"{description} file is empty: {segment_path}")
                        
                        # Use forward slashes for cross-platform compatibility
                        f.write(f"file '{str(segment_path).replace(chr(92), '/')}'\n")
                        print(f"    📎 Added {description}: {segment_path.name}")
                
                print(f"    ✅ Concatenation list created with {len(concat_segments)} segments")
                
            except Exception as e:
                raise Exception(f"Failed to create concatenation list: {str(e)}")
            
            # Concatenate with enhanced error handling
            output_path = VIDEO_TMP_DIR / f"chained_output_{timestamp}.mp4"
            print(f"  Final concatenation to: {output_path.name}")
            
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_list_path),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',  # Fix timing issues
                '-fflags', '+genpts',  # Generate timestamps
                '-an',  # Remove audio
                str(output_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                      timeout=DEFAULT_FFMPEG_TIMEOUT, check=True)
                print("    ✅ Video concatenation completed successfully")
            except subprocess.TimeoutExpired:
                raise Exception(f"Video concatenation timed out after {DEFAULT_FFMPEG_TIMEOUT}s")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else "Unknown FFmpeg error"
                raise Exception(f"Video concatenation failed: {error_msg}")
            
            # Enhanced output validation
            if not output_path.exists():
                raise Exception("Final video file was not created")
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise Exception("Final video file is empty")
            
            # Additional validation using ffprobe
            try:
                validation_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                                '-of', 'default=noprint_wrappers=1:nokey=1', str(output_path)]
                result = subprocess.run(validation_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and result.stdout.strip():
                    duration = float(result.stdout.strip())
                    if duration <= 0:
                        raise Exception("Final video has invalid duration")
                    print(f"    ✅ Output validation passed: {duration:.2f}s duration")
                else:
                    print("    ⚠️ Could not validate video duration, but file exists")
            except Exception as e:
                print(f"    ⚠️ Video validation warning: {e}")
            
            file_size_mb = file_size / (1024 * 1024)
            
            # Calculate total expected duration
            total_duration = sum(info['duration'] for info in video_infos)
            transition_duration = (2 ** num_passes) / 25.0 * 2  # 2 transitions
            expected_duration = total_duration + transition_duration
            
            print(f"\n✅ Chained interpolation complete!")
            print(f"📁 Output: {output_path.name}")
            print(f"📊 Size: {file_size_mb:.1f} MB")
            print(f"⏱️ Expected duration: {expected_duration:.2f}s")
            print(f"🔗 Structure: [Video1] -> [Transition] -> [Video2] -> [Transition] -> [Video3]")
            print(f"🎬 Method: {method.value} with {num_passes} passes")
            
            # Enhanced status message with more details
            status_msg = (
                f"Successfully created chained video ({file_size_mb:.1f} MB). "
                f"Structure: 3 videos with 2 smooth transitions ({num_passes} passes each). "
                f"Expected duration: {expected_duration:.1f}s at {final_fps} FPS."
            )
            
            return str(output_path), status_msg
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise e
        finally:
            # Enhanced cleanup with better retry logic
            if work_dir.exists():
                print("\n🧹 Starting cleanup...")
                cleanup_success = self._safe_rmtree(work_dir)
                if cleanup_success:
                    print("✅ Temporary files cleaned up successfully")
                else:
                    print(f"⚠️ Warning: Could not fully clean up temporary directory: {work_dir}")
                    print("    This may cause disk space issues over time.")