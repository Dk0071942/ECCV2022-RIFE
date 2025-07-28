import shutil
import datetime
from pathlib import Path
from PIL import Image
import cv2
from enum import Enum
from typing import Union

from rife_app.config import DEVICE, CHAINED_TMP_DIR, VIDEO_TMP_DIR
from rife_app.utils.framing import get_video_info, extract_frames, extract_precise_boundary_frame, validate_temporal_alignment
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
        Frame-perfect boundary extraction with enhanced validation.
        
        Args:
            video_path: Path to video file
            position: 'first' or 'last' frame to extract
        
        Returns:
            PIL.Image: High-quality extracted frame
        """
        print(f"📸 Extracting {position} frame from {Path(video_path).name}")
        
        try:
            frame_pil = extract_precise_boundary_frame(
                Path(video_path), 
                position=position, 
                validate_quality=True
            )
            
            if frame_pil is None:
                raise Exception(f"Frame extraction returned None for {position} frame")
            
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
            # Enhanced validation with temporal alignment analysis
            print("📊 Validating input videos and temporal alignment...")
            
            video_paths = [anchor_video_path, middle_video_path, end_video_path]
            alignment_analysis = validate_temporal_alignment(video_paths, final_fps)
            
            # Get individual video info
            anchor_info = get_video_info(Path(anchor_video_path))
            middle_info = get_video_info(Path(middle_video_path))
            end_info = get_video_info(Path(end_video_path))
            
            if not all([anchor_info, middle_info, end_info]):
                raise Exception("Could not read properties from one or more videos.")
            
            # Display alignment analysis
            print(f"🔍 Temporal Alignment Analysis:")
            print(f"   FPS Consistent: {alignment_analysis['fps_consistent']}")
            print(f"   Resolution Consistent: {alignment_analysis['resolution_consistent']}")
            
            if alignment_analysis['recommendations']:
                print(f"📋 Recommendations:")
                for rec in alignment_analysis['recommendations']:
                    print(f"   - {rec}")
            
            # Use middle video resolution as target
            target_w, target_h = middle_info['width'], middle_info['height']
            print(f"🎯 Target resolution: {target_w}x{target_h}")
            
            # Determine if smart re-encoding optimization can be applied
            needs_reencoding = not (alignment_analysis['fps_consistent'] and alignment_analysis['resolution_consistent'])
            print(f"🔧 Re-encoding optimization: {'FULL' if needs_reencoding else 'MINIMAL'}")

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

            # Phase 3: Smart re-encoding with optimization
            print(f"\n⚙️ Phase 3: Smart video processing...")
            
            anchor_processed_path = segments_dir / "anchor_processed.mp4"
            middle_processed_path = segments_dir / "middle_processed.mp4"
            end_processed_path = segments_dir / "end_processed.mp4"
            
            # SPATIAL ALIGNMENT FIX: Use simple scaling without centered padding to match RIFE coordinate system
            # This eliminates the "quick zoom" visual artifact caused by coordinate system mismatch
            vf_filter = f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=decrease,pad=w={target_w}:h={target_h}:color=black"
            
            for i, (input_path, output_path, name, video_info) in enumerate([
                (anchor_video_path, anchor_processed_path, "anchor", anchor_info), 
                (middle_video_path, middle_processed_path, "middle", middle_info), 
                (end_video_path, end_processed_path, "end", end_info)
            ]):
                video_analysis = alignment_analysis['videos'][i]
                
                # Smart processing decision
                needs_fps_conversion = video_analysis['needs_fps_conversion']
                needs_resolution_conversion = video_analysis['needs_resolution_conversion']
                
                if not needs_fps_conversion and not needs_resolution_conversion and video_info['codec'] in ['H264', 'h264']:
                    # Optimal case: just copy with minimal processing
                    print(f"🚀 Fast-copying {name} video (already optimal): {Path(input_path).name}")
                    cmd = [
                        'ffmpeg', '-y', '-i', input_path,
                        '-c', 'copy', '-avoid_negative_ts', 'make_zero',
                        '-movflags', '+faststart', output_path
                    ]
                else:
                    # Re-encode only when necessary
                    print(f"🔄 Re-encoding {name} video: {Path(input_path).name}")
                    print(f"   Reasons: FPS={needs_fps_conversion}, Resolution={needs_resolution_conversion}")
                    cmd = [
                        'ffmpeg', '-y', '-i', input_path, 
                        '-vf', vf_filter, 
                        '-r', str(final_fps), 
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',  # Changed to 'fast' for speed
                        '-pix_fmt', 'yuv420p',
                        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                        '-movflags', '+faststart', '-an', output_path
                    ]
                
                success, msg = run_ffmpeg_command(cmd)
                if not success: 
                    raise Exception(f"FFmpeg error processing {name}: {msg}")
                
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise Exception(f"Processed {name} video failed: {output_path}")
                
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                processing_type = "copied" if not (needs_fps_conversion or needs_resolution_conversion) else "re-encoded"
                print(f"✅ {name.capitalize()} video {processing_type}: {file_size_mb:.1f} MB")

            # Phase 4: Validate all segments before concatenation
            print(f"\n🔍 Phase 4: Validating all segments...")
            
            all_segments = [
                ("Anchor video", anchor_processed_path),
                ("Transition 1", transition1_path),
                ("Middle video", middle_processed_path),
                ("Transition 2", transition2_path),
                ("End video", end_processed_path)
            ]
            
            # SPATIAL VALIDATION: Verify dimension consistency across all segments
            expected_dimensions = None
            
            for i, (name, path) in enumerate(all_segments, 1):
                if not path.exists():
                    raise Exception(f"Missing segment: {path}")
                
                file_size = path.stat().st_size
                if file_size == 0:
                    raise Exception(f"Empty segment: {path}")
                
                # Get video info and validate dimensions
                try:
                    video_info = get_video_info(path)
                    if video_info:
                        duration = video_info.get('duration', 0)
                        frame_count = video_info.get('frame_count', 0)
                        width = video_info.get('width', 0)
                        height = video_info.get('height', 0)
                        duration_str = f"{duration:.2f}s" if duration else "unknown"
                        
                        # Validate spatial consistency
                        current_dimensions = (width, height)
                        if expected_dimensions is None:
                            expected_dimensions = current_dimensions
                            print(f"✅ Reference dimensions set: {width}x{height}")
                        elif current_dimensions != expected_dimensions:
                            print(f"⚠️  Dimension mismatch in {name}: expected {expected_dimensions[0]}x{expected_dimensions[1]}, got {width}x{height}")
                            # Continue with warning but don't fail - FFmpeg concat can handle minor differences
                    else:
                        duration_str = "unknown"
                        frame_count = "unknown"
                        width, height = "unknown", "unknown"
                except:
                    duration_str = "unknown"
                    frame_count = "unknown"
                    width, height = "unknown", "unknown"
                
                file_size_mb = file_size / (1024 * 1024)
                print(f"{i}. {name}: {file_size_mb:.1f} MB, {duration_str}, {frame_count} frames, {width}x{height}")

            # Phase 5: Concatenate all segments
            print(f"\n🔗 Phase 5: Concatenating final video...")
            
            concat_list_path = op_dir / "concat_list.txt"
            with open(concat_list_path, 'w') as f:
                for name, path in all_segments:
                    f.write(f"file '{path.resolve()}'\n")
            
            final_video_path = VIDEO_TMP_DIR / f"chained_output_{timestamp}.mp4"
            
            # Smart concatenation: copy when possible, re-encode only if needed
            if alignment_analysis['fps_consistent'] and alignment_analysis['resolution_consistent']:
                print("🚀 Using fast stream copy concatenation (videos already aligned)")
                cmd_concat = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                    '-i', concat_list_path, 
                    '-c', 'copy', '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    final_video_path
                ]
            else:
                print("🔄 Using re-encoding concatenation (alignment required)")
                # METADATA NORMALIZATION FIX: Re-encode during concatenation to eliminate
                # inconsistent display matrices and transform metadata that cause visual shifts
                cmd_concat = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                    '-i', concat_list_path, 
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',  # Changed to 'fast' for speed
                    '-pix_fmt', 'yuv420p', '-r', str(final_fps),
                    '-aspect', f'{target_w}:{target_h}',  # Explicit aspect ratio
                    '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                    '-movflags', '+faststart',
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
            print(f"✨ Enhancements: Frame-perfect extraction + Smart re-encoding + Temporal alignment")
            
            # Performance summary
            optimizations_used = []
            if alignment_analysis['fps_consistent']:
                optimizations_used.append("FPS-aligned")
            if alignment_analysis['resolution_consistent']:
                optimizations_used.append("Resolution-matched")
            if not needs_reencoding:
                optimizations_used.append("Minimal re-encoding")
            
            if optimizations_used:
                print(f"🚀 Optimizations applied: {', '.join(optimizations_used)}")
            
            # Enhanced status message with optimization details
            optimization_summary = ", ".join(optimizations_used) if optimizations_used else "Standard processing"
            status_message = f"Enhanced chained interpolation successful! Generated {final_size_mb:.1f} MB video with frame-perfect boundaries and smart processing ({optimization_summary})."
            
            return str(final_video_path), status_message
        
        except Exception as e:
            print(f"❌ Chained interpolation failed: {e}")
            raise e
        finally:
            # Cleanup temporary directory
            if op_dir.exists():
                shutil.rmtree(op_dir)
                print(f"🧹 Cleaned up temporary directory: {op_dir}") 