# RIFE Docker Setup

This Docker setup allows you to run the RIFE video interpolation app in a containerized environment.

## Prerequisites

- Docker with GPU support (nvidia-container-toolkit)
- NVIDIA GPU with CUDA support
- At least 4GB GPU memory for optimal performance

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
cd ECCV2022-RIFE
docker-compose up --build
```

### Option 2: Using Docker directly

```bash
cd ECCV2022-RIFE

# Build the image
docker build -t rife-app .

# Run the container
docker run --gpus all -p 7860:7860 -v $(pwd)/temp_gradio:/app/temp_gradio rife-app
```

## Access the Application

Once the container is running, open your browser and navigate to:
http://localhost:7860

## Features Available

The RIFE Docker app provides:

1. **2X Interpolation**: Double the frame rate of videos
2. **4X Interpolation**: Quadruple the frame rate of videos  
3. **NX Interpolation**: Custom frame rate multiplication
4. **Simple Video Re-encoding**: Convert video formats
5. **Image Interpolation**: Create intermediate frames between two images

## Volume Mounts

- `./temp_gradio:/app/temp_gradio` - Temporary files for processing

## GPU Requirements

- **Minimum**: 4GB VRAM for HD video processing
- **Recommended**: 6GB+ VRAM for optimal performance
- **CPU Fallback**: Will work on CPU but significantly slower

## Troubleshooting

### Container fails to start
- Ensure nvidia-container-toolkit is installed
- Check GPU availability with `nvidia-smi`

### Out of memory errors
- Reduce video resolution or frame count
- Close other GPU-intensive applications

### FFmpeg not found
- The Dockerfile includes FFmpeg installation
- If issues persist, rebuild the container

## Performance Notes

- RTX 2080 Ti: ~30+ FPS for 2X 720p interpolation
- Processing time scales with video resolution and length
- GPU acceleration provides 10-50x speedup over CPU processing