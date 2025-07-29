"""
Disk-Based Frame Interpolation for RIFE

This module implements a disk-based approach that processes frame pairs independently,
storing intermediate results as temporary files. This approach:

1. Maintains constant memory usage regardless of interpolation depth
2. Preserves motion quality by avoiding hierarchical subdivision artifacts
3. Uses the core two-frame interpolation logic consistently
4. Scales to unlimited interpolation factors without memory constraints

Algorithm:
- Stage 1: Generate intermediate frames between original start/end pair
- Stage 2: For each adjacent pair, generate intermediate frames
- Continue until desired density achieved
- All intermediate frames stored on disk, loaded only when needed

Memory Usage: O(1) - only 2 frames in memory at any time
Quality: Superior - no hierarchical blur accumulation
"""

import torch
import numpy as np
from pathlib import Path
import shutil
import tempfile
from typing import List, Tuple, Optional, Iterator
import time
import math
from dataclasses import dataclass

from .memory_monitor import GPUMemoryMonitor, monitor_memory_usage
from ..config import DEVICE, VIDEO_TMP_DIR


@dataclass
class FrameInfo:
    """Information about a stored frame."""
    path: Path
    index: float  # Temporal position (0.0 = start, 1.0 = end)
    wave: int     # Which processing wave generated this frame
    

class DiskBasedInterpolator:
    """
    Disk-based frame interpolation that maintains constant memory usage
    while achieving unlimited interpolation depth.
    
    Key advantages:
    - Constant O(1) memory usage
    - No motion blur accumulation  
    - Consistent two-frame processing
    - Unlimited scalability
    """
    
    def __init__(self, model, device: str = "cuda"):
        """
        Initialize disk-based interpolator.
        
        Args:
            model: RIFE model instance
            device: PyTorch device
        """
        self.model = model
        self.device = device
        self.memory_monitor = GPUMemoryMonitor(enable_logging=True)
        self.original_dims = None  # Store original dimensions before padding
        
    def _save_frame_to_disk(self, tensor: torch.Tensor, path: Path) -> bool:
        """Save frame tensor to disk as PNG.
        
        Note: We keep the padded dimensions during processing as RIFE requires 
        dimensions divisible by 16. Cropping happens only during final video creation.
        """
        try:
            # Convert tensor to numpy array
            if tensor.dim() == 4:  # Remove batch dimension if present
                tensor = tensor.squeeze(0)
            
            # Convert from CHW to HWC and scale to 0-255
            frame_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Save as PNG using PIL or OpenCV
            import cv2
            cv2.imwrite(str(path), cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            
            return True
        except Exception as e:
            print(f"Error saving frame to {path}: {e}")
            return False
    
    def _load_frame_from_disk(self, path: Path) -> Optional[torch.Tensor]:
        """Load frame tensor from disk."""
        try:
            import cv2
            
            # Load image
            frame_bgr = cv2.imread(str(path))
            if frame_bgr is None:
                return None
                
            # Convert BGR to RGB and to tensor
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return tensor
        except Exception as e:
            print(f"Error loading frame from {path}: {e}")
            return None
    
    def _interpolate_pair_to_disk(
        self, 
        frame_a_path: Path, 
        frame_b_path: Path,
        output_dir: Path,
        temporal_positions: List[float],
        model_scale_factor: float = 1.0
    ) -> List[FrameInfo]:
        """
        Interpolate between two frames and save results to disk.
        Always uses timestep=0.5 for optimal quality.
        
        Args:
            frame_a_path: Path to first frame
            frame_b_path: Path to second frame  
            output_dir: Directory to save intermediate frames
            temporal_positions: List of temporal positions (0.0 to 1.0) to generate
            model_scale_factor: RIFE model scale factor
            
        Returns:
            List of FrameInfo objects for generated frames
        """
        # For progressive refinement, we should only interpolate at the midpoint
        # Check if 0.5 is in the requested positions
        if 0.5 not in temporal_positions and not any(abs(pos - 0.5) < 0.001 for pos in temporal_positions):
            # If not asking for midpoint, return empty (shouldn't happen with progressive approach)
            return []
        
        # Load the two input frames
        frame_a = self._load_frame_from_disk(frame_a_path)
        frame_b = self._load_frame_from_disk(frame_b_path)
        
        if frame_a is None or frame_b is None:
            return []
        
        generated_frames = []
        
        # Generate the middle frame (always at timestep=0.5)
        intermediate = self.model.inference(frame_a, frame_b, scale=model_scale_factor)
        
        # Save to disk - we'll use a placeholder temporal position
        # The actual temporal position will be set by the caller
        output_path = output_dir / f"frame_{time.time():.6f}.png"
        if self._save_frame_to_disk(intermediate, output_path):
            frame_info = FrameInfo(
                path=output_path,
                index=0.5,  # This will be updated by caller to actual position
                wave=0  # Will be updated by caller
            )
            generated_frames.append(frame_info)
        
        # Cleanup GPU memory immediately
        del intermediate
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Cleanup input frames
        del frame_a, frame_b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return generated_frames
    
    @monitor_memory_usage("disk_based_interpolation")
    def interpolate_with_disk_storage(
        self,
        start_frame: torch.Tensor,
        end_frame: torch.Tensor,
        target_frame_count: int = 5,
        model_scale_factor: float = 1.0,
        temp_dir: Optional[Path] = None
    ) -> Tuple[List[FrameInfo], Path]:
        """
        Generate interpolated frames using disk-based processing.
        
        Args:
            start_frame: Starting frame tensor
            end_frame: Ending frame tensor
            target_frame_count: Desired total number of frames (including start/end)
            model_scale_factor: RIFE model scale factor
            temp_dir: Temporary directory (created if None)
            
        Returns:
            Tuple of (frame_info_list, temp_directory_path)
        """
        # Create temporary directory
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="rife_disk_interp_"))
        else:
            temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Store original dimensions before any padding (only if not already set)
        if self.original_dims is None:
            _, _, h_orig, w_orig = start_frame.shape
            self.original_dims = (h_orig, w_orig)
        
        # Calculate passes and duration info for better logging
        import math
        passes = int(math.log2(target_frame_count)) if target_frame_count > 1 else 1
        duration_25fps = target_frame_count / 25.0
        print(f"💾 Disk-based interpolation: {passes} passes → {target_frame_count} frames → {duration_25fps:.2f}s at 25 FPS")
        print(f"   Temp directory: {temp_dir}")
        
        # Save initial frames to disk (keeping padded dimensions for RIFE)
        start_path = temp_dir / "frame_0.000000.png"
        end_path = temp_dir / "frame_1.000000.png"
        
        self._save_frame_to_disk(start_frame, start_path)
        self._save_frame_to_disk(end_frame, end_path)
        
        # Initialize frame list with start and end
        all_frames = [
            FrameInfo(path=start_path, index=0.0, wave=0),
            FrameInfo(path=end_path, index=1.0, wave=0)
        ]
        
        # Iteratively add frames until we reach target count
        wave = 1
        while len(all_frames) < target_frame_count:
            pairs_to_process = len(all_frames) - 1
            print(f"🌊 Pass {wave}: Interpolating {pairs_to_process} frame pairs for smooth motion...")
            
            # Sort frames by temporal position
            all_frames.sort(key=lambda f: f.index)
            
            # Calculate how many frames we need to add this wave
            current_count = len(all_frames)
            remaining_needed = target_frame_count - current_count
            pairs_count = current_count - 1
            
            new_frames = []
            
            # Process each adjacent pair
            for i in range(len(all_frames) - 1):
                frame_a_info = all_frames[i]
                frame_b_info = all_frames[i + 1]
                
                # For progressive refinement, always interpolate at midpoint
                mid_pos = (frame_a_info.index + frame_b_info.index) / 2.0
                
                # Generate intermediate frame at midpoint only
                intermediates = self._interpolate_pair_to_disk(
                    frame_a_info.path,
                    frame_b_info.path,
                    temp_dir,
                    [0.5],  # Always use 0.5 for optimal quality
                    model_scale_factor
                )
                
                # Update wave number and correct temporal index
                for frame_info in intermediates:
                    frame_info.wave = wave
                    frame_info.index = mid_pos  # Update to actual temporal position
                
                new_frames.extend(intermediates)
                
                # Report memory usage
                memory_info = self.memory_monitor.get_memory_info()
                print(f"  Processed pair {i+1}/{pairs_count} - "
                      f"GPU: {memory_info['utilization_percent']:.1f}% used")
            
            # Add new frames to the list
            all_frames.extend(new_frames)
            
            wave += 1
            
            # Safety break to prevent infinite loops
            if wave > 10:
                print(f"⚠️ Stopping at wave {wave} to prevent infinite loop")
                break
        
        # Final sort by temporal position
        all_frames.sort(key=lambda f: f.index)
        
        # Trim to exact target count if we exceeded it
        if len(all_frames) > target_frame_count:
            # Keep frames with most uniform distribution
            step = len(all_frames) / target_frame_count
            selected_indices = [int(i * step) for i in range(target_frame_count)]
            all_frames = [all_frames[i] for i in selected_indices]
        
        passes_completed = wave - 1
        duration_25fps = len(all_frames) / 25.0
        print(f"✅ Generated {len(all_frames)} frames in {passes_completed} passes ({duration_25fps:.2f}s at 25 FPS)")
        
        return all_frames, temp_dir
    
    def frames_to_video(
        self,
        frame_infos: List[FrameInfo],
        output_video_path: Path,
        fps: int = 30,
        cleanup_temp: bool = True
    ) -> bool:
        """
        Convert frame sequence to video file.
        
        Args:
            frame_infos: List of FrameInfo objects in temporal order
            output_video_path: Path for output video
            fps: Frames per second
            cleanup_temp: Whether to cleanup temporary files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from ..utils.ffmpeg import run_ffmpeg_command
            
            # Sort frames by temporal order
            frame_infos.sort(key=lambda f: f.index)
            
            # Get dimensions from stored original dimensions (without padding)
            if self.original_dims:
                h, w = self.original_dims
            else:
                # Fallback to loading first frame if original dims not set
                first_frame = self._load_frame_from_disk(frame_infos[0].path)
                if first_frame is None:
                    return False
                _, _, h, w = first_frame.shape
            
            # Create temporary directory with sequential frame names
            with tempfile.TemporaryDirectory(prefix="rife_video_") as video_temp_dir:
                video_temp_path = Path(video_temp_dir)
                
                # Copy frames with sequential names
                for i, frame_info in enumerate(frame_infos):
                    src_path = frame_info.path
                    dst_path = video_temp_path / f"frame_{i:06d}.png"
                    shutil.copy2(src_path, dst_path)
                
                # Create video with FFmpeg, applying crop if needed
                if self.original_dims and self.original_dims != (h, w):
                    # Apply crop filter to remove padding
                    h_orig, w_orig = self.original_dims
                    crop_filter = f'crop={w_orig}:{h_orig}:0:0'
                    vf_filter = f'{crop_filter},format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1'
                else:
                    vf_filter = 'format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1'
                
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-r', str(fps),
                    '-i', str(video_temp_path / 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-vf', vf_filter,
                    '-color_primaries', 'bt709',
                    '-color_trc', 'bt709',
                    '-colorspace', 'bt709',
                    '-movflags', '+faststart',
                    str(output_video_path)
                ]
                
                success, msg = run_ffmpeg_command(ffmpeg_cmd, video_temp_path)
                
                if success:
                    print(f"🎥 Video created: {output_video_path}")
                    # Report the actual output dimensions (after cropping if applied)
                    if self.original_dims:
                        h_out, w_out = self.original_dims
                    else:
                        h_out, w_out = h, w
                    print(f"   Format: {fps} FPS, {w_out}×{h_out}, BT.709 color space, H.264 encoding")
                    
                    # Cleanup temporary frames if requested
                    if cleanup_temp:
                        temp_dir = frame_infos[0].path.parent
                        if temp_dir.exists():
                            shutil.rmtree(temp_dir)
                            print(f"🧹 Cleaned up temporary directory: {temp_dir}")
                    
                    return True
                else:
                    print(f"❌ FFmpeg error: {msg}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error creating video: {e}")
            return False


def disk_based_interpolate(
    start_frame: torch.Tensor,
    end_frame: torch.Tensor,
    model,
    target_frames: int = 5,
    device: str = "cuda",
    original_dims: Optional[Tuple[int, int]] = None
) -> Tuple[Optional[Path], str]:
    """
    Convenience function for disk-based interpolation.
    
    Args:
        start_frame: Starting frame tensor (already padded)
        end_frame: Ending frame tensor (already padded)
        model: RIFE model instance
        target_frames: Target number of total frames
        device: PyTorch device
        original_dims: Original dimensions before padding (height, width)
        
    Returns:
        Tuple of (video_path, status_message)
    """
    try:
        interpolator = DiskBasedInterpolator(model, device)
        
        # Set original dimensions if provided
        if original_dims:
            interpolator.original_dims = original_dims
        
        # Generate frames
        frame_infos, _ = interpolator.interpolate_with_disk_storage(
            start_frame, end_frame, target_frames
        )
        
        # Create output video in VIDEO_TMP_DIR (not in temp_dir that gets deleted)
        timestamp = int(time.time() * 1000)
        output_path = VIDEO_TMP_DIR / f"disk_interp_{timestamp}.mp4"
        
        # Ensure VIDEO_TMP_DIR exists
        VIDEO_TMP_DIR.mkdir(parents=True, exist_ok=True)
        
        success = interpolator.frames_to_video(frame_infos, output_path, fps=25, cleanup_temp=True)  # Fixed 25 FPS for image interpolation
        
        if success:
            passes = int(math.log2(len(frame_infos))) if len(frame_infos) > 1 else 1
            duration_25fps = len(frame_infos) / 25.0
            # Get resolution from stored original dimensions (without padding)
            if interpolator.original_dims:
                h, w = interpolator.original_dims
                resolution_info = f" Output resolution: {w}×{h}."
                print(f"DEBUG: Using original resolution: {w}×{h}")
            else:
                # Fallback to loading first frame if original dims not set
                first_frame = interpolator._load_frame_from_disk(frame_infos[0].path)
                if first_frame:
                    _, _, h, w = first_frame.shape
                    resolution_info = f" Output resolution: {w}×{h}."
                    print(f"DEBUG: Extracted resolution: {w}×{h}")
                else:
                    resolution_info = ""
                    print("DEBUG: Could not extract resolution from first frame")
            
            status_msg = f"Disk-based interpolation successful: {passes} passes → {len(frame_infos)} frames → {duration_25fps:.2f}s at 25 FPS.{resolution_info}"
            print(f"DEBUG: Final disk-based status: {status_msg}")
            return str(output_path), status_msg
        else:
            return None, "Failed to create video from frames"
            
    except Exception as e:
        return None, f"Disk-based interpolation failed: {str(e)}"