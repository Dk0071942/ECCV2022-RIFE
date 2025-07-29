# Video Interpolation and Loop Creation Toolkit

This toolkit provides a set of Python scripts for video processing, frame interpolation, and loop creation. It combines the power of RIFE (Real-time Intermediate Flow Estimation) for frame interpolation with custom tools for video analysis and loop creation.

## Features

- **Frame Interpolation**: High-quality frame interpolation using RIFE
- **Loop Creation**: Create seamless video loops with neutral frame transitions
- **Video Connection**: Connect two videos with smooth RIFE interpolation transitions
- **Frame Analysis**: Find similar frames in videos for loop creation
- **Video Processing**: Extract and process video segments with precise frame control

## Prerequisites

- Python 3.7 or higher
- FFmpeg installed and available in system PATH
- CUDA-capable GPU (recommended for RIFE interpolation)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Interpolate.git
cd Interpolate
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

4. Download the RIFE model files:
   - Place the model files in the `model/` directory

## Scripts Overview

### 1. `find_frames.py`
Finds similar frames in a video that can be used for creating loops.

**Usage:**
```bash
python find_frames.py
```
The script will prompt you for:
- Video file path
- Output directory
- Frame interval
- Minimum gap between matches
- SSIM threshold
- Number of similar frames to find
- Optional reference frame number

### 2. `core_video_erstellen_cli.py`
Creates video loops and extracts segments from videos.

**Usage:**
```bash
# For frame extraction:
python core_video_erstellen_cli.py input_video.mp4 output_folder --fps 30 extract --segments "100-200,300-400"

# For loop creation:
python core_video_erstellen_cli.py input_video.mp4 output_folder --fps 30 loop --start_frame "500" --durations "60,120,180"
```

### 3. `advanced_video_looper.py`
Creates advanced video loops using RIFE interpolation with neutral frame transitions, or connects two videos with smooth interpolated transitions.

**Loop Mode Usage:**
```bash
# Create looped videos with neutral frame transitions
python advanced_video_looper.py loop neutral_loop.mp4 source_clips_folder output_folder --rife_exp 2

# With additional options
python advanced_video_looper.py loop neutral_loop.mp4 source_clips_folder output_folder \
    --rife_exp 3 --workers 8 --output_fps 60 --skip_incompatible --resume
```

**Connect Mode Usage (NEW):**
```bash
# Basic connection with audio transfer from first video
python advanced_video_looper.py connect video1.mp4 video2.mp4 connected_output.mp4

# Connection without audio
python advanced_video_looper.py connect video1.mp4 video2.mp4 connected_output.mp4 --no_audio

# Custom interpolation settings
python advanced_video_looper.py connect video1.mp4 video2.mp4 output.mp4 \
    --rife_exp 3 --output_fps 60
```

**Connect Mode Options:**
- `video1`: Path to the first video (required)
- `video2`: Path to the second video (required)
- `output`: Path for the connected output video (required)
- `--rife_exp`: RIFE exponent for interpolation (default: 2, higher = more frames)
- `--output_fps`: Output video frame rate (default: 30.0)
- `--no_audio`: Don't transfer audio from the first video
- `--quality`: Quality mode - "fast", "balanced", or "high_quality" (affects frame extraction quality)
- `--preprocess`: Enable frame preprocessing to reduce artifacts (noise reduction, sharpening, color normalization)
- `--denoise`: Apply additional denoising before interpolation (slower but can reduce artifacts)

**Improving Interpolation Quality:**

If you experience artifacts in the interpolated frames, try these options:

```bash
# For high-quality results (slower but better quality)
python advanced_video_looper.py connect video1.mp4 video2.mp4 output.mp4 \
    --quality high_quality --preprocess

# For green screen or noisy content
python advanced_video_looper.py connect video1.mp4 video2.mp4 output.mp4 \
    --preprocess --denoise

# For more interpolated frames (smoother transition)
python advanced_video_looper.py connect video1.mp4 video2.mp4 output.mp4 \
    --rife_exp 4 --quality high_quality --preprocess
```

**Quality Settings Explained:**
- `--quality fast`: Lower extraction quality but faster processing
- `--quality balanced`: Good balance of quality and speed (default)
- `--quality high_quality`: Best extraction quality with advanced denoising (slower)
- `--rife_exp`: Higher values create more intermediate frames (2=3 frames, 3=7 frames, 4=15 frames)
- `--preprocess`: Enables noise reduction, sharpening, and color normalization
- `--denoise`: Additional denoising specifically for artifact reduction

**Common Artifact Solutions:**
- **Motion blur artifacts**: Use `--rife_exp 3` or `--rife_exp 4` and `--quality high_quality`
- **Green screen halos**: Use `--preprocess` (automatically detects and handles green screens)
- **Color inconsistencies**: Use `--preprocess` for color normalization
- **Noisy/grainy results**: Use `--denoise` and `--quality high_quality`
- **Choppy transitions**: Increase `--rife_exp` to 3 or 4 for more intermediate frames

### 4. `inference_video.py`
Performs frame interpolation on video files.

**Usage:**
```bash
python inference_video.py --video input_video.mp4 --exp 1 --fps 60
```

### 5. `inference_img.py`
Performs frame interpolation between pairs of images.

**Usage:**
```bash
python inference_img.py --img img0.png img1.png --exp 4
```

## Directory Structure

```
Interpolate/
├── advanced_video_looper.py
├── benchmark/
├── core_video_erstellen_cli.py
├── dataset.py
├── demo/
├── docker/
├── find_frames.py
├── inference_img.py
├── inference_video.py
├── model/
├── output/
├── train.py
└── train_log/
```

## Common Issues and Solutions

1. **FFmpeg not found**:
   - Ensure FFmpeg is installed and added to your system PATH
   - Verify installation with `ffmpeg -version`

2. **CUDA/GPU issues**:
   - Check CUDA installation: `nvidia-smi`
   - Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Memory issues**:
   - Reduce batch size or frame resolution
   - Process videos in smaller segments

4. **File path issues**:
   - When specifying paths with spaces or special characters, make sure to use quotes
   - For network paths, ensure you have proper access permissions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) for the frame interpolation model
- FFmpeg for video processing capabilities 

# Video Frame Pair Finder

This Python script analyzes a video file to find pairs of frames that are highly similar to each other based on a combination of visual content, human pose, and facial features. It's designed to identify suitable start and end frames for tasks like video segment interpolation or finding recurring scenes/poses.

## Features

*   **Flexible Frame Sampling**: Loads frames from the video at a user-defined interval.
*   **Multi-Stage Similarity Assessment**:
    *   **Structural Similarity (SSIM)**: Initial comparison based on overall image similarity (luminance, contrast, structure).
    *   **Pose Estimation (Hands/Face Focus)**: Utilizes MediaPipe Pose to compare the positions of critical body landmarks (primarily hands, face, and shoulders).
    *   **Facial Landmark Comparison**: Employs MediaPipe Face Mesh for a detailed comparison of 468 facial landmarks, ensuring facial expression and structure are similar.
*   **Configurable Constraints**:
    *   Minimum and maximum frame distance (gap) between frames in a pair.
    *   Adjustable similarity thresholds for SSIM, pose, and facial features.
    *   Option to limit the number of output pairs or save all found pairs.
*   **Targeted Search**:
    *   Option to specify a particular video frame number as the start of pairs to search for.
    *   Alternatively, search through the entire video for all qualifying pairs.
*   **Output**:
    *   Saves the identified start and end frames of each matched pair as PNG images.
    *   Provides a console summary listing the details of each found pair (indices, similarity scores).

## Prerequisites

*   Python 3.7+
*   FFmpeg (often required by OpenCV for video processing, usually handled during OpenCV installation or system setup).

## Installation

1.  **Clone the repository or download the `find_frames.py` script.**
2.  **Install the required Python libraries:**
    Navigate to the script's directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Execute the script from your terminal:
    ```bash
    python find_frames.py
    ```
2.  The script will then prompt you for various settings:

    *   **Pfad zur Videodatei**: The full path to your input video file.
    *   **Pfad zum Ausgabeordner**: Directory where matched frame images will be saved.
    *   **Frame-Intervall**: Analyze every Nth frame (e.g., 5 means analyze frame 0, 5, 10...).
    *   **Mindestabstand zwischen Frames eines Paares**: Minimum number of original video frames that must separate the start and end frame of a pair (e.g., 30).
    *   **Maximaler Abstand zwischen Frames eines Paares**: Maximum number of original video frames allowed between the start and end frame (e.g., 600 for 20s @30fps). Enter 0 for no limit.
    *   **SSIM-Schwellenwert**: Similarity threshold for SSIM (0.0 to 1.0, e.g., 0.98). Higher is more similar.
    *   **Max. Anzahl zu speichernder Frame-Paare**: Maximum number of pairs to save (e.g., 10). Enter 0 to save all found pairs.
    *   **Pose-Filter (Hände/Gesicht) aktivieren?**: (ja/nein) Enable comparison of critical pose landmarks.
        *   *If ja*: Pose-Ähnlichkeitsschwellenwert (e.g., 0.05). Lower is more similar.
        *   *If ja*: Sichtbarkeitsschwellenwert für Keypoints (e.g., 0.5).
        *   *If ja*: Min. kritischer Keypoints für Vergleich (e.g., 5).
    *   **Gesichtsmerkmale-Filter aktivieren?**: (ja/nein) Enable detailed facial landmark comparison.
        *   *If ja*: Gesichts-Ähnlichkeitsschwellenwert (e.g., 0.01). Lower is more similar.
    *   **Spezifische Start-Frame-Nummer für Paare**: Enter a video frame number to only find pairs starting with this frame. Leave blank to search the entire video.

## Output

*   **Image Files**: For each identified pair, two PNG images (`PAIR_S<start_idx>_E<end_idx>_START_SSIM<score>.png` and `PAIR_S<start_idx>_E<end_idx>_END_SSIM<score>.png`) are saved in the specified output directory.
*   **Console Summary**: A list of found pairs is printed to the console, including their start and end video frame indices, SSIM score, and (if enabled) pose and face difference scores.

This script provides a powerful way to automate the search for visually and structurally similar segments in a video, tailored to human subjects. 