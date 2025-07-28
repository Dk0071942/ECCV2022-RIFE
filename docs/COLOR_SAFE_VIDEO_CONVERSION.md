# Color-Safe Video ↔ PNG Frame Conversion Guide

## The Problem: Color Shift in Video Frame Extraction

When extracting frames from video to PNG and rebuilding, many users experience color shifts, typically resulting in darker or washed-out videos. This happens because:

1. **Range Mismatch**: Videos use limited range (16-235), PNGs use full range (0-255)
2. **Color Space Confusion**: Incorrect handling of BT.709 vs sRGB transfer functions
3. **Multiple Conversions**: Each YUV↔RGB conversion can introduce quantization errors

## The Solution: Color-Safe Round-Trip

This guide provides a proven method to extract video frames to PNG and rebuild without any color shift.

### Key Principles

1. **Single Conversion Each Way**: Convert only once when extracting, once when rebuilding
2. **Explicit Range Handling**: Always specify input and output ranges explicitly
3. **Proper Metadata**: Set correct color metadata flags for player compatibility

## Implementation

### Method 1: Using Standard FFmpeg (No zscale Required)

This method works with any FFmpeg installation:

#### Step 1: Extract Frames (Video → PNG)

```bash
ffmpeg -y -i input.mp4 \
       -vf "scale=in_range=limited:out_range=full,format=rgb24" \
       -color_range pc \
       frame_%06d.png
```

**What happens:**
- `scale=in_range=limited:out_range=full` - Expands limited range (16-235) to full range (0-255)
- `format=rgb24` - Ensures RGB output format
- `-color_range pc` - Marks PNGs as full range

#### Step 2: Rebuild Video (PNG → Video)

```bash
ffmpeg -y -r 25 -start_number 1 -i frame_%06d.png -i input.mp4 \
       -vf "scale=in_range=full:out_range=limited,format=yuv420p" \
       -c:v libx264 -crf 18 -preset slow \
       -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
       -map 0:v -map 1:a? -c:a copy \
       -movflags +faststart \
       output.mp4
```

**What happens:**
- `scale=in_range=full:out_range=limited` - Compresses full range (0-255) back to limited (16-235)
- `format=yuv420p` - Standard video pixel format
- `-color_range tv` - Marks video as limited range
- Color metadata flags ensure proper playback

### Method 2: Using zscale (If Available)

If your FFmpeg has zscale support (compiled with libzimg), this provides even more precise control:

#### Step 1: Extract Frames (Video → PNG)

```bash
ffmpeg -y -i input.mp4 \
       -vf "zscale=range=full:primaries=bt709:transfer=srgb,format=rgb24" \
       frame_%06d.png
```

#### Step 2: Rebuild Video (PNG → Video)

```bash
ffmpeg -y -r 25 -start_number 1 -i frame_%06d.png -i input.mp4 \
       -vf "zscale=range=limited:primaries=bt709:transfer=bt709,format=yuv420p" \
       -c:v libx264 -crf 18 -preset slow \
       -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
       -map 0:v -map 1:a? -c:a copy \
       -movflags +faststart \
       output.mp4
```

## Practical Examples

### Example 1: Basic Frame Extraction and Rebuild

```bash
# Extract frames from a 30fps video
ffmpeg -i myvideo.mp4 \
       -vf "scale=in_range=limited:out_range=full,format=rgb24" \
       -color_range pc \
       frames/frame_%06d.png

# Do your editing on the PNG files...

# Rebuild at original framerate
ffmpeg -r 30 -i frames/frame_%06d.png -i myvideo.mp4 \
       -vf "scale=in_range=full:out_range=limited,format=yuv420p" \
       -c:v libx264 -crf 18 -preset slow \
       -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
       -map 0:v -map 1:a? -c:a copy \
       output.mp4
```

### Example 2: Frame Extraction for AI Processing

```bash
# Extract every 5th frame for AI upscaling
ffmpeg -i source.mp4 \
       -vf "select='not(mod(n,5))',scale=in_range=limited:out_range=full,format=rgb24" \
       -vsync vfr -color_range pc \
       selected_frames/frame_%06d.png

# Process with AI upscaler...

# Rebuild with processed frames
ffmpeg -r 25 -i processed_frames/frame_%06d.png -i source.mp4 \
       -vf "scale=in_range=full:out_range=limited,format=yuv420p" \
       -c:v libx264 -crf 18 -preset slow \
       -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
       -map 0:v -map 1:a? -c:a copy \
       upscaled.mp4
```

### Example 3: Maintaining Audio Sync

```bash
# Extract with timestamps preserved
ffmpeg -i input.mp4 \
       -vf "scale=in_range=limited:out_range=full,format=rgb24" \
       -color_range pc \
       -vsync passthrough \
       frames/frame_%06d.png

# Rebuild maintaining original timing
ffmpeg -r 29.97 -i frames/frame_%06d.png -i input.mp4 \
       -vf "scale=in_range=full:out_range=limited,format=yuv420p" \
       -c:v libx264 -crf 18 -preset slow \
       -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
       -map 0:v -map 1:a -c:a copy \
       -shortest \
       output.mp4
```

## Verification

### Check Your Results

After rebuilding, verify the color metadata:

```bash
ffprobe -hide_banner -show_streams -select_streams v output.mp4 | grep -E 'color_'
```

You should see:
```
color_range=tv
color_space=bt709
color_transfer=bt709
color_primaries=bt709
```

### Visual Comparison

```bash
# Extract first frame from original and output
ffmpeg -i input.mp4 -vf "scale=in_range=limited:out_range=full" -vframes 1 original.png
ffmpeg -i output.mp4 -vf "scale=in_range=limited:out_range=full" -vframes 1 rebuilt.png

# Compare visually or with ImageMagick
compare -metric RMSE original.png rebuilt.png difference.png
```

## Common Issues and Solutions

### Issue: FFmpeg doesn't have zscale
**Solution**: Use the standard scale filter method shown above

### Issue: Colors still look different
**Check**:
1. Verify input video color range: `ffprobe -show_streams input.mp4 | grep color_range`
2. If `color_range=pc` (full range), adjust extraction: `scale=in_range=full:out_range=full`
3. Some videos may have incorrect metadata - trust your eyes

### Issue: PNG files look dark in image viewers
**This is normal!** Full-range PNGs may appear different in image editors that assume sRGB gamma. The important thing is they convert back correctly.

### Issue: Banding or posterization
**Solution**: Use higher quality settings:
- Extraction: Add `-compression_level 0` for lossless PNGs
- Rebuild: Use `-crf 16` or lower for higher quality

## Advanced Tips

### 10-bit Processing

For 10-bit sources or higher quality:

```bash
# Extract to 16-bit PNGs
ffmpeg -i input.mp4 \
       -vf "scale=in_range=limited:out_range=full,format=rgb48" \
       -color_range pc \
       frame_%06d.png

# Rebuild to 10-bit
ffmpeg -r 25 -i frame_%06d.png -i input.mp4 \
       -vf "scale=in_range=full:out_range=limited,format=yuv420p10le" \
       -c:v libx264 -crf 18 -preset slow \
       -pix_fmt yuv420p10le \
       -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
       -map 0:v -map 1:a? -c:a copy \
       output_10bit.mp4
```

### Batch Processing Script

```bash
#!/bin/bash
# color_safe_extract.sh

INPUT="$1"
OUTPUT_DIR="${2:-frames}"
mkdir -p "$OUTPUT_DIR"

echo "Extracting frames from $INPUT..."
ffmpeg -i "$INPUT" \
       -vf "scale=in_range=limited:out_range=full,format=rgb24" \
       -color_range pc \
       "$OUTPUT_DIR/frame_%06d.png"

echo "Frames extracted to $OUTPUT_DIR/"
echo "To rebuild: ./color_safe_rebuild.sh $OUTPUT_DIR output.mp4 $INPUT"
```

## Python Implementation

```python
import subprocess
import os
from pathlib import Path

def extract_frames(video_path, output_dir, use_zscale=False):
    """Extract video frames to PNG with color-safe conversion."""
    Path(output_dir).mkdir(exist_ok=True)
    
    if use_zscale:
        vf = "zscale=range=full:primaries=bt709:transfer=srgb,format=rgb24"
    else:
        vf = "scale=in_range=limited:out_range=full,format=rgb24"
    
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', vf,
        '-color_range', 'pc',
        f"{output_dir}/frame_%06d.png"
    ]
    
    subprocess.run(cmd, check=True)

def rebuild_video(frames_dir, output_path, original_video, fps=25, use_zscale=False):
    """Rebuild video from PNG frames with color-safe conversion."""
    
    if use_zscale:
        vf = "zscale=range=limited:primaries=bt709:transfer=bt709,format=yuv420p"
    else:
        vf = "scale=in_range=full:out_range=limited,format=yuv420p"
    
    cmd = [
        'ffmpeg', '-y',
        '-r', str(fps),
        '-i', f"{frames_dir}/frame_%06d.png",
        '-i', original_video,  # For audio
        '-vf', vf,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
        '-color_range', 'tv',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-map', '0:v', '-map', '1:a?',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        output_path
    ]
    
    subprocess.run(cmd, check=True)

# Example usage
if __name__ == "__main__":
    extract_frames("input.mp4", "frames")
    # Process frames here...
    rebuild_video("frames", "output.mp4", "input.mp4", fps=30)
```

## Conclusion

By following this guide, you can confidently extract video frames to PNG and rebuild them without any color shift. The key is proper range conversion and explicit metadata handling. Whether you're doing AI processing, manual editing, or automated workflows, these methods ensure your colors remain accurate throughout the process.

### Remember:
- Always convert range explicitly (limited ↔ full)
- Set proper color metadata flags
- Use the same method consistently
- Verify your results

Happy frame processing! 🎬