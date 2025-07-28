# Color-Safe Video ↔ PNG Conversion - Quick Reference

## 🎯 The One-Line Solution

### Extract Frames (Video → PNG)
```bash
ffmpeg -i input.mp4 -vf "scale=in_range=limited:out_range=full,format=rgb24" -color_range pc frame_%06d.png
```

### Rebuild Video (PNG → Video)
```bash
ffmpeg -r 25 -i frame_%06d.png -i input.mp4 -vf "scale=in_range=full:out_range=limited,format=yuv420p" -c:v libx264 -crf 18 -preset slow -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 -map 0:v -map 1:a? -c:a copy -movflags +faststart output.mp4
```

## 📋 Copy-Paste Templates

### Standard Quality (CRF 18)
```bash
# Extract
ffmpeg -i VIDEO -vf "scale=in_range=limited:out_range=full,format=rgb24" -color_range pc frames/f_%06d.png

# Rebuild  
ffmpeg -r FPS -i frames/f_%06d.png -i VIDEO -vf "scale=in_range=full:out_range=limited,format=yuv420p" -c:v libx264 -crf 18 -preset slow -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 -map 0:v -map 1:a? -c:a copy -movflags +faststart OUTPUT
```

### High Quality (CRF 16, Lossless PNG)
```bash
# Extract
ffmpeg -i VIDEO -vf "scale=in_range=limited:out_range=full,format=rgb24" -compression_level 0 -color_range pc frames/f_%06d.png

# Rebuild
ffmpeg -r FPS -i frames/f_%06d.png -i VIDEO -vf "scale=in_range=full:out_range=limited,format=yuv420p" -c:v libx264 -crf 16 -preset slow -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 -map 0:v -map 1:a? -c:a copy -movflags +faststart OUTPUT
```

### 10-bit Processing
```bash
# Extract to 16-bit PNG
ffmpeg -i VIDEO -vf "scale=in_range=limited:out_range=full,format=rgb48" -color_range pc frames/f_%06d.png

# Rebuild to 10-bit
ffmpeg -r FPS -i frames/f_%06d.png -i VIDEO -vf "scale=in_range=full:out_range=limited,format=yuv420p10le" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p10le -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 -map 0:v -map 1:a? -c:a copy OUTPUT
```

## 🔍 Quick Checks

### Verify Color Metadata
```bash
ffprobe -show_streams -select_streams v output.mp4 | grep -E 'color_'
```

Should show:
```
color_range=tv
color_space=bt709
color_transfer=bt709
color_primaries=bt709
```

### Compare First Frames
```bash
# Extract first frame from both videos
ffmpeg -i original.mp4 -vf "scale=in_range=limited:out_range=full" -vframes 1 frame_orig.png
ffmpeg -i rebuilt.mp4 -vf "scale=in_range=limited:out_range=full" -vframes 1 frame_new.png
```

## ⚡ Key Points

1. **Always specify ranges**: `in_range=limited:out_range=full` (extract) and vice versa (rebuild)
2. **Set color metadata**: `-color_range pc` for PNG, `-color_range tv` for video
3. **Include all color flags**: `-color_primaries bt709 -color_trc bt709 -colorspace bt709`
4. **Map audio from original**: `-i original.mp4 -map 0:v -map 1:a?`

## 🚨 Common Mistakes to Avoid

❌ **Don't** use default FFmpeg extraction:
```bash
ffmpeg -i video.mp4 frame_%d.png  # Will cause color shift!
```

❌ **Don't** forget color metadata:
```bash
ffmpeg -i frames/f_%06d.png output.mp4  # Missing color flags!
```

❌ **Don't** process multiple times:
```bash
video → png → video → png → video  # Each conversion loses quality!
```

✅ **Do** process once and keep originals:
```bash
original.mp4 → frames → processed_frames → final.mp4
```

## 🎬 Ready-to-Use Scripts

### extract_frames.sh
```bash
#!/bin/bash
ffmpeg -i "$1" -vf "scale=in_range=limited:out_range=full,format=rgb24" -color_range pc "${2:-frames}/frame_%06d.png"
```

### rebuild_video.sh
```bash
#!/bin/bash
FPS="${3:-25}"
ffmpeg -r $FPS -i "$1/frame_%06d.png" -i "$4" -vf "scale=in_range=full:out_range=limited,format=yuv420p" -c:v libx264 -crf 18 -preset slow -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 -map 0:v -map 1:a? -c:a copy -movflags +faststart "$2"
```

Usage:
```bash
./extract_frames.sh input.mp4 frames_dir
# Edit frames...
./rebuild_video.sh frames_dir output.mp4 25 input.mp4
```