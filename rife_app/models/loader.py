import torch
from rife_app.config import MODEL_DIR

# Global model variable
loaded_model = None

def get_model():
    global loaded_model
    if loaded_model is not None:
        return loaded_model

    # Try to load the model (same logic as inference_img.py and inference_video.py)
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                m = Model()
                m.load_model(str(MODEL_DIR), -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model # Common case from user logs
                m = Model()
                m.load_model(str(MODEL_DIR), -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            m = Model()
            m.load_model(str(MODEL_DIR), -1)
            print("Loaded v1.x HD model")
        loaded_model = m
    except Exception as e:
        try:
            from model.RIFE import Model # Fallback to arxiv RIFE
            m = Model()
            m.load_model(str(MODEL_DIR), -1)
            print("Loaded ArXiv-RIFE model")
            loaded_model = m
        except Exception as e_arxiv:
            error_message = f"Error loading any RIFE model. Main error: {e}, Arxiv fallback error: {e_arxiv}. Check MODEL_DIR ('{MODEL_DIR}') and ensure model files (e.g., RIFE_HDv3.py) are in the correct location (e.g., 'train_log' or 'model' directory) and importable."
            print(error_message)
            raise RuntimeError(error_message) # Stop app launch if model fails

    loaded_model.eval()
    loaded_model.device()
    return loaded_model

def setup_torch_device():
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU mode.") 