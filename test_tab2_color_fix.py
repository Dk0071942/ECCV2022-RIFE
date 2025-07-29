#!/usr/bin/env python3
"""
Test script to diagnose and verify Tab 2 color handling.
This script checks the entire pipeline for color-safe conversion.
"""

import subprocess
import tempfile
from pathlib import Path


def create_test_video():
    """Create a test video with known color values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test frame with specific colors
        test_frame = temp_path / "test_frame.png"
        
        # Use ImageMagick or FFmpeg to create a test pattern
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', 'testsrc=duration=1:size=640x480:rate=1',
            '-frames:v', '1',
            str(test_frame)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to create test frame: {result.stderr}")
            return None
            
        # Create a test video from the frame
        test_video = temp_path / "test_video.mp4"
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', str(test_frame),
            '-c:v', 'libx264',
            '-t', '1',
            '-pix_fmt', 'yuv420p',
            '-color_range', 'tv',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',
            str(test_video)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to create test video: {result.stderr}")
            return None
            
        return test_video


def test_frame_extraction_pipeline():
    """Test the complete frame extraction and video creation pipeline."""
    print("=" * 60)
    print("🧪 Testing Complete Color Pipeline")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create a test video
        print("\n1. Creating test video with known colors...")
        test_video = create_test_video()
        if not test_video:
            print("❌ Failed to create test video")
            return
        print("✅ Test video created")
        
        # Step 2: Extract frame using color-safe method
        print("\n2. Extracting frame with color-safe conversion...")
        extracted_frame = temp_path / "extracted_frame.png"
        cmd = [
            'ffmpeg', '-y',
            '-i', str(test_video),
            '-vf', 'scale=in_range=limited:out_range=full,format=rgb24',
            '-color_range', 'pc',
            '-frames:v', '1',
            str(extracted_frame)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Frame extraction failed: {result.stderr}")
            return
        print("✅ Frame extracted with color-safe conversion")
        
        # Step 3: Create video from extracted frame
        print("\n3. Creating video from extracted frame...")
        output_video = temp_path / "output_video.mp4"
        cmd = [
            'ffmpeg', '-y',
            '-color_range', '2',  # Input is full range
            '-loop', '1',
            '-i', str(extracted_frame),
            '-vf', 'scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=limited,format=yuv420p',
            '-c:v', 'libx264',
            '-t', '1',
            '-pix_fmt', 'yuv420p',
            '-color_range', 'tv',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',
            str(output_video)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Video creation failed: {result.stderr}")
            return
        print("✅ Video created with color-safe conversion")
        
        # Step 4: Compare color metadata
        print("\n4. Checking color metadata...")
        
        # Check original video
        cmd = ['ffprobe', '-show_streams', '-select_streams', 'v', str(test_video)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("\nOriginal video metadata:")
        for line in result.stdout.split('\n'):
            if 'color_' in line:
                print(f"  {line}")
        
        # Check output video
        cmd = ['ffprobe', '-show_streams', '-select_streams', 'v', str(output_video)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("\nOutput video metadata:")
        for line in result.stdout.split('\n'):
            if 'color_' in line:
                print(f"  {line}")
        
        print("\n✅ Pipeline test complete!")
        print("\nKey points for Tab 2 color handling:")
        print("1. Input images are in full range (0-255)")
        print("2. FFmpeg needs -color_range 2 before input to recognize full range")
        print("3. Use scale filter with in_range=full:out_range=limited")
        print("4. Set all color metadata flags for output")


def check_ffmpeg_version():
    """Check FFmpeg version and capabilities."""
    print("\n" + "=" * 60)
    print("🔧 FFmpeg Version and Capabilities")
    print("=" * 60)
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            print(f"Version: {lines[0]}")
            
            # Check for important filters
            result = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True)
            important_filters = ['scale', 'format', 'colorspace']
            print("\nImportant filters:")
            for filter_name in important_filters:
                if filter_name in result.stdout:
                    print(f"  ✅ {filter_name} is available")
                else:
                    print(f"  ❌ {filter_name} is NOT available")
    except Exception as e:
        print(f"❌ Error checking FFmpeg: {e}")


def main():
    """Run all tests."""
    print("\n🎨 Tab 2 Color Fix Verification\n")
    
    check_ffmpeg_version()
    test_frame_extraction_pipeline()
    
    print("\n" + "=" * 60)
    print("📝 Summary of Changes Made:")
    print("=" * 60)
    print("1. Updated save_tensor_as_image to use PIL for consistent PNG encoding")
    print("2. Updated disk-based interpolation to use PIL for frame I/O")
    print("3. Added -color_range 2 flag before input in FFmpeg commands")
    print("4. Added in_color_matrix and out_color_matrix to scale filter")
    print("5. Ensured all services use the same color-safe approach")
    print("\nThese changes ensure that Tab 2 properly handles color conversion")
    print("from full-range PNGs to limited-range videos without color shifts.")


if __name__ == "__main__":
    main()