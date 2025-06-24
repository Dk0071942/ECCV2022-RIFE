import sys
from pathlib import Path
import shutil

# Ensure the rife_app package is importable
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from rife_app.models.loader import get_model, setup_torch_device
from rife_app.config import DEVICE
from rife_app.services.video_interpolator import VideoInterpolator

def main_interpolate(
    input_video_path: str,
    output_dir_path: str,
    exp: int = 1,
    use_fp16: bool = False,
):
    """
    Wrapper for video interpolation using RIFE services.
    Generates an interpolated video and saves it in output_dir_path.
    """
    # Prepare torch/device settings
    setup_torch_device()
    model = get_model()
    interpolator = VideoInterpolator(model, DEVICE)

    # Perform interpolation with default scale factors and no UI progress
    video_path_temp, status = interpolator.interpolate(
        input_video_path=input_video_path,
        interpolation_exp=exp,
        model_scale_factor=1.0,
        output_res_scale_factor=1.0,
        target_fps_override=0,
        use_fp16=use_fp16,
        progress=lambda *args, **kwargs: None,
    )

    # Print status for visibility
    print(status)

    # Move the generated video to the desired output directory
    temp_path = Path(video_path_temp)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / temp_path.name
    shutil.move(str(temp_path), str(dest_path))

    return str(dest_path) 