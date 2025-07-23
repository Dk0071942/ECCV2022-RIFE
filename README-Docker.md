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

# Run the container with proper volume mounts
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/temp_gradio:/app/temp_gradio \
  -v $(pwd)/train_log:/app/train_log:ro \
  --restart unless-stopped \
  rife-app
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
- `./train_log:/app/train_log:ro` - RIFE model files (read-only mount)

## Improved Features

### Following LatentSync Best Practices
- **Optimized layer caching**: Dependencies installed before code copy
- **Model verification**: Automatic validation of RIFE model files
- **Health checking**: Built-in container health monitoring
- **Production ready**: Proper restart policies and resource management

### Performance Optimizations
- **Efficient builds**: .dockerignore reduces build context size
- **Layer optimization**: Better Docker layer caching for faster rebuilds
- **Resource management**: Proper GPU allocation and memory handling

## GPU Requirements

- **Minimum**: 4GB VRAM for HD video processing
- **Recommended**: 6GB+ VRAM for optimal performance
- **CPU Fallback**: Will work on CPU but significantly slower

## Troubleshooting

### Container fails to start
- Ensure nvidia-container-toolkit is installed
- Check GPU availability with `nvidia-smi`
- Verify Docker has GPU support: `docker run --gpus all nvidia/cuda:12.6-runtime-ubuntu22.04 nvidia-smi`

### Module not found errors (gradio, pillow, etc.)
- Updated requirements.txt includes all necessary dependencies
- Rebuild the container: `docker-compose build --no-cache`

### Out of memory errors
- Reduce video resolution or frame count
- Close other GPU-intensive applications
- Try running with CPU fallback by setting environment variable: `-e CUDA_VISIBLE_DEVICES=""`

### FFmpeg not found
- The Dockerfile includes FFmpeg installation
- If issues persist, rebuild the container

### Deprecated CUDA image warnings
- Updated to use nvidia/cuda:12.6-cudnn-devel-ubuntu22.04 (current stable version)
- If warnings persist, they're informational and won't affect functionality

## Performance Notes

- RTX 2080 Ti: ~30+ FPS for 2X 720p interpolation
- Processing time scales with video resolution and length
- GPU acceleration provides 10-50x speedup over CPU processing