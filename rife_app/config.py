from pathlib import Path
import torch

# Paths
BASE_DIR         = Path(__file__).resolve().parent.parent
MODEL_DIR        = BASE_DIR / "train_log"
TEMP_DIR          = BASE_DIR / "temp_gradio"
IMAGE_TMP_DIR    = TEMP_DIR / "images"
VIDEO_TMP_DIR    = TEMP_DIR / "videos"
CHAINED_TMP_DIR  = TEMP_DIR / "chained"

# Defaults
DEFAULT_FPS      = 25
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure temporary directories exist
TEMP_DIR.mkdir(exist_ok=True)
IMAGE_TMP_DIR.mkdir(exist_ok=True)
VIDEO_TMP_DIR.mkdir(exist_ok=True)
CHAINED_TMP_DIR.mkdir(exist_ok=True) 