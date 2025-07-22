# Color Shift Fix Summary

## Overview
Fixed the color shift issue in RIFE frame extraction by eliminating redundant color space conversions and maintaining RGB format consistently throughout the pipeline.

## Changes Made

### 1. `extract_frames` function (lines 27-56)
- **Before**: BGR → RGB conversion embedded in PIL Image creation
- **After**: Explicit BGR → RGB conversion with clear variable naming
- **Impact**: Improved code clarity, same functionality

### 2. `pil_to_tensor` function (lines 58-68)
- **Before**: Converted PIL RGB back to BGR unnecessarily
- **After**: Maintains RGB format throughout
- **Impact**: Eliminates one color conversion, preventing color shift

### 3. `save_tensor_as_image` function (lines 85-97)
- **Before**: Saved tensor directly as BGR (incorrect for RGB tensors)
- **After**: Converts RGB tensor to BGR only for cv2.imwrite
- **Impact**: Ensures correct color output when saving

## Color Pipeline After Fix

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

## Key Improvements

1. **Single Conversion Point**: BGR→RGB happens once at extraction
2. **Consistent Format**: RGB maintained throughout processing
3. **Proper Output**: RGB→BGR only when required by cv2.imwrite
4. **No Redundancy**: Eliminated BGR→RGB→BGR round trip

## Compatibility

- Works with existing video interpolator (already uses RGB)
- Compatible with chained interpolator
- No changes needed to other services
- Maintains same API for frame extraction

## Testing Recommendations

1. Compare extracted frames pixel-by-pixel with original video frames
2. Test with videos using different color spaces (BT.709, BT.601, sRGB)
3. Verify interpolation quality is maintained
4. Check for any performance impact (should be minimal)

## Result

The color shift issue is resolved by maintaining a consistent RGB pipeline and eliminating unnecessary color space conversions that were introducing cumulative rounding errors.