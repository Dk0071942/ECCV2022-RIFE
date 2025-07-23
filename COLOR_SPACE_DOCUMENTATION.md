# RIFE Color Space Documentation

Complete documentation covering color space handling, fixes, and pipeline in the RIFE interpolation system.

## Overview

The RIFE interpolation system processes video frames through a complex color space pipeline. This document covers the key issues identified, fixes implemented, and the final color space handling approach.

## Color Space Pipeline

### Current Correct Pipeline (Post-Fix)

```
Video (YUV/BGR in file)
    ↓ OpenCV VideoCapture
BGR Frame Data
    ↓ cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) [Once only]
RGB Frame Data
    ↓ Image.fromarray()
PIL Image (RGB)
    ↓ np.array() [No conversion]
RGB NumPy Array
    ↓ torch.from_numpy().transpose()
RGB Tensor
    ↓ Processing...
RGB Tensor Output
    ↓ cv2.cvtColor(array, cv2.COLOR_RGB2BGR) [Only for saving]
BGR for cv2.imwrite
```

## Historical Issues and Fixes

### Issue 1: Color Shift in Frame Extraction

**Problem**: Redundant color space conversions in the frame extraction pipeline caused visible color shifts.

**Root Cause**: The pipeline had a BGR → RGB → BGR round trip conversion:
- OpenCV VideoCapture provides BGR frames
- Conversion to RGB for PIL Image processing
- Conversion back to BGR for tensor creation
- Each conversion introduced rounding errors

**Code Locations Affected**:
- `rife_app/utils/framing.py:39` - BGR to RGB conversion
- `rife_app/utils/framing.py:52` - RGB back to BGR conversion  
- `rife_app/utils/framing.py:78` - Direct BGR save

**Solution**: Eliminated redundant conversions by maintaining RGB format consistently throughout the processing pipeline.

**Changes Made**:
1. **extract_frames** - Explicit BGR → RGB conversion with clear variable naming
2. **pil_to_tensor** - Maintains RGB format throughout (no BGR conversion)
3. **save_tensor_as_image** - Converts RGB tensor to BGR only for cv2.imwrite

### Issue 2: Missing BT.709 Color Space Metadata

**Problem**: Generated videos lacked proper BT.709 color space metadata, causing validation failures during re-encoding.

**Missing Metadata**:
- Color Primaries: missing (expected bt709)
- Color Trc: missing (expected bt709)
- Colorspace: missing (expected bt709)

**Services Fixed**:

#### ImageInterpolator Service (`rife_app/services/image_interpolator.py`)
Added complete BT.709 metadata to FFmpeg command:
```python
ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-r', str(fps),
    '-i', frames_dir / 'frame_%05d.png',
    '-s', f'{w}x{h}',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-crf', '18',
    '-pix_fmt', 'yuv420p',
    '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1',
    '-color_primaries', 'bt709',
    '-color_trc', 'bt709',
    '-colorspace', 'bt709',
    '-movflags', '+faststart',
    output_video_path
]
```

#### ChainedInterpolator Service (`rife_app/services/chained.py`)
Added BT.709 metadata to segment generation:
```python
cmd = [
    'ffmpeg', '-y', '-r', str(fps), '-i', frames_dir / 'frame_%05d.png',
    '-s', f'{w}x{h}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1',
    '-color_primaries', 'bt709',
    '-color_trc', 'bt709',
    '-colorspace', 'bt709',
    '-movflags', '+faststart', segment_path
]
```

## Current Status

### ✅ Properly Configured Services

1. **VideoInterpolator Service** - Complete BT.709 setup implemented
2. **SimpleVideoReencoder Service** - Complete BT.709 setup implemented
3. **ImageInterpolator Service** - Fixed with complete BT.709 metadata
4. **ChainedInterpolator Service** - Fixed with complete BT.709 metadata

### ⚠️ Known Limitations

**Main Inference Script (`inference_video.py`)**:
- Uses `cv2.VideoWriter` which doesn't support color space metadata
- Consider replacing with FFmpeg subprocess for critical workflows
- Alternative: Add post-processing step to inject metadata

## Testing and Validation

### Color Accuracy Testing
1. Compare extracted frames pixel-by-pixel with original video frames
2. Test with videos using different color spaces (BT.709, BT.601, sRGB)
3. Verify interpolation quality is maintained
4. Check for performance impact (should be minimal)

### Color Space Validation
Generated videos now pass validation with:
- ✅ Color Primaries: bt709
- ✅ Color Trc: bt709
- ✅ Colorspace: bt709

## Technical Details

### Color Space Conversion Best Practices
1. **Single Conversion Point**: BGR→RGB happens once at extraction
2. **Consistent Format**: RGB maintained throughout processing
3. **Proper Output**: RGB→BGR only when required by cv2.imwrite
4. **No Redundancy**: Eliminated BGR→RGB→BGR round trips

### Performance Impact
- **Color Fix**: No performance impact - eliminates redundant conversions
- **Metadata Fix**: No performance impact - same FFmpeg parameters with metadata

### Compatibility
- Works with all existing video interpolator services
- Compatible with chained interpolator
- Maintains same API for frame extraction
- Full backward compatibility maintained

## Implementation Summary

### Files Modified
- `rife_app/services/image_interpolator.py` - Added BT.709 metadata
- `rife_app/services/chained.py` - Added BT.709 metadata
- `rife_app/utils/framing.py` - Fixed color conversion pipeline

### Key Improvements
1. **Eliminated Color Shift**: Removed redundant color conversions
2. **Added Metadata**: All video outputs include proper BT.709 color space metadata
3. **Improved Pipeline**: Consistent RGB format throughout processing
4. **Maintained Compatibility**: No API changes, full backward compatibility

### Coverage
- All Gradio UI video outputs now have proper color space handling
- Frame extraction maintains color accuracy
- Video generation includes proper metadata
- Interpolation quality preserved while fixing color issues