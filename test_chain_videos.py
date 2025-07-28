#!/usr/bin/env python3
"""
Test script for chained video interpolation.
Uses 1.mp4 as start and end, 2.mp4 as middle.
Saves interpolated frames as separate videos for debugging.
"""

import sys
import shutil
import datetime
import cv2
from pathlib import Path
from PIL import Image

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rife_app.config import CHAINED_TMP_DIR, VIDEO_TMP_DIR
from rife_app.models.loader import get_model
from rife_app.services.image_interpolator import ImageInterpolator
from rife_app.utils.ffmpeg import run_ffmpeg_command


def extract_frames(video_path, output_dir):
    """Extract all frames from a video."""
    print(f"\n📸 Extracting frames from {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_path = output_dir / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)
        frame_idx += 1
    
    cap.release()
    print(f"✅ Extracted {len(frames)} frames at {fps} FPS")
    return frames, fps


def get_frame_as_pil(frame_path):
    """Load frame as PIL Image."""
    frame_bgr = cv2.imread(str(frame_path))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def save_interpolated_frames_as_video(frames_dir, output_path, fps=25):
    """Save frames directory as video."""
    # Get first frame to determine dimensions
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise Exception("No frames found")
    
    first_frame = cv2.imread(str(frames[0]))
    height, width = first_frame.shape[:2]
    
    # Create video
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-r', str(fps),
        '-i', str(frames_dir / 'frame_%05d.png'),
        '-s', f'{width}x{height}',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(output_path)
    ]
    
    success, msg = run_ffmpeg_command(ffmpeg_cmd)
    if not success:
        raise Exception(f"FFmpeg error: {msg}")
    
    return output_path


def test_chain_videos():
    """Test chained interpolation with 1.mp4 as start/end and 2.mp4 as middle."""
    
    # Setup paths
    video1_path = Path("1.mp4")
    video2_path = Path("2.mp4")
    
    if not video1_path.exists() or not video2_path.exists():
        print("❌ Error: 1.mp4 and 2.mp4 must exist in current directory")
        return
    
    print("🎬 Starting chained video test")
    print(f"📹 Using {video1_path} as start and end")
    print(f"📹 Using {video2_path} as middle")
    
    # Load model
    print("\n🤖 Loading RIFE model...")
    try:
        model = get_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load RIFE model: {e}")
        return
    
    # Create image interpolator
    interpolator = ImageInterpolator(model)
    
    # Create working directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = CHAINED_TMP_DIR / f"test_chain_{timestamp}"
    work_dir.mkdir(parents=True)
    
    try:
        # Extract frames from videos
        video1_frames_dir = work_dir / "video1_frames"
        video2_frames_dir = work_dir / "video2_frames"
        video1_frames_dir.mkdir()
        video2_frames_dir.mkdir()
        
        frames1, fps1 = extract_frames(video1_path, video1_frames_dir)
        frames2, fps2 = extract_frames(video2_path, video2_frames_dir)
        
        # Get boundary frames
        print("\n🎯 Getting boundary frames...")
        last_frame_video1 = get_frame_as_pil(frames1[-1])
        first_frame_video2 = get_frame_as_pil(frames2[0])
        last_frame_video2 = get_frame_as_pil(frames2[-1])
        first_frame_video1 = get_frame_as_pil(frames1[0])  # For end->start transition
        
        print(f"Video 1 last frame: {frames1[-1].name}")
        print(f"Video 2 first frame: {frames2[0].name}")
        print(f"Video 2 last frame: {frames2[-1].name}")
        print(f"Video 1 first frame: {frames1[0].name}")
        
        # Create transition 1: video1 -> video2
        print("\n🎨 Creating transition 1: video1 -> video2")
        transition1_path, status1 = interpolator.interpolate(
            img0_pil=last_frame_video1,
            img1_pil=first_frame_video2,
            num_passes=2,
            fps=25,
            use_disk_based=True
        )
        
        if not transition1_path:
            raise Exception(f"Transition 1 failed: {status1}")
        
        print(f"✅ Transition 1 created: {transition1_path}")
        
        # Create transition 2: video2 -> video1
        print("\n🎨 Creating transition 2: video2 -> video1")
        transition2_path, status2 = interpolator.interpolate(
            img0_pil=last_frame_video2,
            img1_pil=first_frame_video1,
            num_passes=2,
            fps=25,
            use_disk_based=True
        )
        
        if not transition2_path:
            raise Exception(f"Transition 2 failed: {status2}")
        
        print(f"✅ Transition 2 created: {transition2_path}")
        
        # Save transitions to output directory
        output_dir = Path(".")
        shutil.copy2(transition1_path, output_dir / f"transition1_{timestamp}.mp4")
        shutil.copy2(transition2_path, output_dir / f"transition2_{timestamp}.mp4")
        
        print(f"\n✅ Saved transition videos:")
        print(f"   - transition1_{timestamp}.mp4 (video1 -> video2)")
        print(f"   - transition2_{timestamp}.mp4 (video2 -> video1)")
        
        # Extract and save interpolated frames for debugging
        print("\n🔍 Extracting interpolated frames for debugging...")
        
        # Extract transition 1 frames
        trans1_frames_dir = work_dir / "transition1_frames"
        trans1_frames_dir.mkdir()
        extract_frames(transition1_path, trans1_frames_dir)
        
        # Extract transition 2 frames  
        trans2_frames_dir = work_dir / "transition2_frames"
        trans2_frames_dir.mkdir()
        extract_frames(transition2_path, trans2_frames_dir)
        
        # Create final chained video
        print("\n🔗 Creating final chained video...")
        all_frames_dir = work_dir / "all_frames"
        all_frames_dir.mkdir()
        
        frame_counter = 0
        
        # Copy video1 frames (excluding last)
        for frame in frames1[:-1]:
            shutil.copy2(frame, all_frames_dir / f"frame_{frame_counter:07d}.png")
            frame_counter += 1
        
        # Copy transition1 frames
        for frame in sorted(trans1_frames_dir.glob("*.png")):
            shutil.copy2(frame, all_frames_dir / f"frame_{frame_counter:07d}.png")
            frame_counter += 1
        
        # Copy video2 frames (excluding first and last)
        for frame in frames2[1:-1]:
            shutil.copy2(frame, all_frames_dir / f"frame_{frame_counter:07d}.png")
            frame_counter += 1
        
        # Copy transition2 frames
        for frame in sorted(trans2_frames_dir.glob("*.png")):
            shutil.copy2(frame, all_frames_dir / f"frame_{frame_counter:07d}.png")
            frame_counter += 1
        
        # Copy video1 frames again (excluding first, since we want to end where we started)
        for frame in frames1[1:]:
            shutil.copy2(frame, all_frames_dir / f"frame_{frame_counter:07d}.png")
            frame_counter += 1
        
        print(f"Total frames combined: {frame_counter}")
        
        # Create final video
        final_output = output_dir / f"chained_output_{timestamp}.mp4"
        save_interpolated_frames_as_video(all_frames_dir, final_output, fps=25)
        
        print(f"\n✅ Created final chained video: chained_output_{timestamp}.mp4")
        print(f"\n📊 Summary:")
        print(f"   - Video 1 frames: {len(frames1)}")
        print(f"   - Video 2 frames: {len(frames2)}")
        print(f"   - Transition frames: ~4 each (2 passes)")
        print(f"   - Total output frames: {frame_counter}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir)
            print("\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    test_chain_videos()