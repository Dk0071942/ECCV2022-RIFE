# Tab 2 Color Fix Explanation

## The Problem

Tab 2 was still showing color shifts even after implementing color-safe conversion. The issue was multi-faceted:

1. **Input Images**: When images come from Tab 1's video extraction, they are already in full range (0-255) due to the color-safe extraction
2. **PNG Saving**: The tensor-to-PNG conversion wasn't consistently handling color encoding
3. **FFmpeg Input Recognition**: FFmpeg wasn't properly recognizing that input PNGs were in full range

## The Solution

### 1. Consistent PNG Handling

Updated both `save_tensor_as_image` and disk-based interpolation to use PIL for saving PNGs:

```python
# Save using PIL to ensure proper PNG encoding
from PIL import Image
img_pil = Image.fromarray(img_to_save_uint8)
img_pil.save(str(path), 'PNG', compress_level=0)  # Lossless PNG
```

This ensures all PNGs are saved consistently in full range (0-255), which is the standard for PNG files.

### 2. FFmpeg Input Color Range Specification

Added explicit input color range specification to all FFmpeg commands:

```bash
ffmpeg -y \
  -color_range 2 \  # Specify input is full range (2 = pc/full)
  -r 25 \
  -i frame_%05d.png \
  -vf "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=limited,format=yuv420p" \
  ...
```

Key changes:
- `-color_range 2` before input tells FFmpeg the PNGs are full range
- Added `in_color_matrix=bt709:out_color_matrix=bt709` to the scale filter
- This ensures proper color matrix handling during conversion

### 3. Complete Color Metadata

All video creation commands now include:
- `-color_range tv` - Output is limited range
- `-color_primaries bt709` - BT.709 color primaries
- `-color_trc bt709` - BT.709 transfer characteristics
- `-colorspace bt709` - BT.709 color space

## Why This Works

1. **PNG Standard**: PNGs are always full range (0-255). By explicitly telling FFmpeg this, we avoid misinterpretation
2. **Color Matrix**: Specifying both input and output color matrices ensures proper conversion
3. **Metadata Flags**: Complete metadata ensures players correctly interpret the video

## Testing

To verify the fix works:

1. Extract frames from a video using Tab 1
2. Use those frames in Tab 2 for interpolation
3. Compare the output video colors with the original
4. Check metadata with: `ffprobe -show_streams output.mp4 | grep color_`

The colors should now match perfectly without any shifts or changes in brightness/contrast.