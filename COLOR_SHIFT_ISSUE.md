# Color Shift Issue in RIFE Frame Extraction

## Issue Description
When extracting frames from videos using the RIFE app's frame selection feature, the extracted frames exhibit a slight but noticeable color shift compared to the original video frames. This affects the quality and accuracy of frame interpolation.

## Root Cause Analysis

### Color Space Conversion Chain
The issue stems from redundant color space conversions in the frame extraction pipeline:

```
Original Video (YUV) 
    ↓ OpenCV VideoCapture
BGR Frame Data
    ↓ cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
RGB Frame Data 
    ↓ Image.fromarray()
PIL Image (RGB)
    ↓ cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
BGR Array Data
    ↓ torch.from_numpy().transpose()
Tensor (BGR channels)
```

### Specific Code Locations

1. **Frame Extraction** (`rife_app/utils/framing.py`):
   ```python
   # Line 39: BGR to RGB conversion
   pil_start_frame = Image.fromarray(cv2.cvtColor(frame_start_bgr, cv2.COLOR_BGR2RGB))
   ```

2. **PIL to Tensor Conversion** (`rife_app/utils/framing.py`):
   ```python
   # Line 52: RGB back to BGR conversion
   img_cv_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
   ```

3. **Tensor Saving** (`rife_app/utils/framing.py`):
   ```python
   # Line 78: Direct BGR save
   cv2.imwrite(str(path), img_to_save_uint8)
   ```

## Technical Issues

### 1. Redundant Conversions
- BGR → RGB → BGR round trip is unnecessary
- Each conversion introduces small rounding errors
- Floating-point precision loss accumulates

### 2. Color Space Inconsistency
- OpenCV uses BGR natively
- PIL uses RGB natively
- Tensors store BGR but interface expects RGB
- FFmpeg output enforces BT.709 color space

### 3. Precision Loss
- Color values: 0-255 integers → 0-1 floats → 0-255 integers
- Each conversion can introduce ±1 errors per channel
- Cumulative effect creates visible color shift

## Impact

1. **Visual Quality**: Subtle but perceptible color differences
2. **Interpolation Accuracy**: Color inconsistencies between frames
3. **User Experience**: Extracted frames don't match video appearance
4. **Downstream Effects**: Interpolated frames inherit color shift

## Solution Overview

### Approach 1: Eliminate PIL Intermediate (Recommended)
- Extract frames directly as NumPy arrays
- Keep BGR format throughout pipeline
- Create tensors directly from BGR data
- No color space conversions needed

### Approach 2: Consistent RGB Pipeline
- Convert to RGB once at extraction
- Keep RGB throughout processing
- Convert to BGR only for final save
- Minimize conversion points

### Implementation Plan

1. Modify `extract_frames` to return NumPy arrays instead of PIL images
2. Update `pil_to_tensor` to accept NumPy arrays
3. Ensure consistent color space handling
4. Preserve original video color metadata
5. Test with various video formats

## Testing Considerations

- Compare pixel values before/after extraction
- Test with different video color spaces (BT.709, BT.601, etc.)
- Verify no visual differences in extracted frames
- Ensure compatibility with existing interpolation pipeline