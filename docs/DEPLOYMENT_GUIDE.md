# RIFE Deployment Guide

## Overview

This guide covers all deployment methods for the ECCV2022-RIFE system, including Docker containerization, local installation, and production deployment strategies.

## Docker Deployment (Recommended)

### Prerequisites

- Docker with GPU support (nvidia-container-toolkit)
- NVIDIA GPU with CUDA support
- At least 4GB GPU memory for optimal performance

### Quick Start

#### Option 1: Using Docker Compose (Recommended)

```bash
cd ECCV2022-RIFE
docker-compose up --build
```

#### Option 2: Using Docker directly

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

### Access the Application

Once the container is running, open your browser and navigate to:
http://localhost:7860

### Features Available

The RIFE Docker app provides:

1. **2X Interpolation**: Double the frame rate of videos
2. **4X Interpolation**: Quadruple the frame rate of videos  
3. **NX Interpolation**: Custom frame rate multiplication
4. **Simple Video Re-encoding**: Convert video formats
5. **Image Interpolation**: Create intermediate frames between two images

### Volume Mounts

- `./temp_gradio:/app/temp_gradio` - Temporary files for processing
- `./train_log:/app/train_log:ro` - RIFE model files (read-only mount)

### Improved Features

#### Following LatentSync Best Practices
- **Optimized layer caching**: Dependencies installed before code copy
- **Model verification**: Automatic validation of RIFE model files
- **Health checking**: Built-in container health monitoring
- **Production ready**: Proper restart policies and resource management

#### Performance Optimizations
- **Efficient builds**: .dockerignore reduces build context size
- **Layer optimization**: Better Docker layer caching for faster rebuilds
- **Resource management**: Proper GPU allocation and memory handling

### GPU Requirements

- **Minimum**: 4GB VRAM for HD video processing
- **Recommended**: 6GB+ VRAM for optimal performance
- **CPU Fallback**: Will work on CPU but significantly slower

### Troubleshooting

#### Container fails to start
- Ensure nvidia-container-toolkit is installed
- Check GPU availability with `nvidia-smi`
- Verify Docker has GPU support: `docker run --gpus all nvidia/cuda:12.6-runtime-ubuntu22.04 nvidia-smi`

#### Module not found errors (gradio, pillow, etc.)
- Updated requirements.txt includes all necessary dependencies
- Rebuild the container: `docker-compose build --no-cache`

#### Out of memory errors
- Reduce video resolution or frame count
- Close other GPU-intensive applications
- Try running with CPU fallback by setting environment variable: `-e CUDA_VISIBLE_DEVICES=""`

#### FFmpeg not found
- The Dockerfile includes FFmpeg installation
- If issues persist, rebuild the container

#### Deprecated CUDA image warnings
- Updated to use nvidia/cuda:12.6-cudnn-devel-ubuntu22.04 (current stable version)
- If warnings persist, they're informational and won't affect functionality

### Performance Notes

- RTX 2080 Ti: ~30+ FPS for 2X 720p interpolation
- Processing time scales with video resolution and length
- GPU acceleration provides 10-50x speedup over CPU processing

## Local Installation

### Installation

```bash
git clone git@github.com:megvii-research/ECCV2022-RIFE.git
cd ECCV2022-RIFE
pip3 install -r requirements.txt
```

### Model Download

* Download the pretrained **HD** models from [here](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing). (百度网盘链接:https://pan.baidu.com/share/init?surl=u6Q7-i4Hu4Vx9_5BJibPPA 密码:hfk3，把压缩包解开后放在 train_log/\*)

* Unzip and move the pretrained parameters to train_log/\*

* This model is not reported by our paper, for our paper model please refer to [evaluation](https://github.com/hzwer/ECCV2022-RIFE#evaluation).

### Run

**Video Frame Interpolation**

You can use our [demo video](https://drive.google.com/file/d/1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc/view?usp=sharing) or your own video. 

```bash
python3 inference_video.py --exp=1 --video=video.mp4 
```
(generate video_2X_xxfps.mp4)

```bash
python3 inference_video.py --exp=2 --video=video.mp4
```
(for 4X interpolation)

```bash
python3 inference_video.py --exp=1 --video=video.mp4 --scale=0.5
```
(If your video has very high resolution such as 4K, we recommend set --scale=0.5 (default 1.0). If you generate disordered pattern on your videos, try set --scale=2.0. This parameter control the process resolution for optical flow model.)

```bash
python3 inference_video.py --exp=2 --img=input/
```
(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)

```bash
python3 inference_video.py --exp=2 --video=video.mp4 --fps=60
```
(add slomo effect, the audio will be removed)

```bash
python3 inference_video.py --video=video.mp4 --montage --png
```
(if you want to montage the origin video and save the png format output)

**Extended Application**

You may refer to [#278](https://github.com/megvii-research/ECCV2022-RIFE/issues/278#event-7199085190) for **Optical Flow Estimation** and refer to [#291](https://github.com/hzwer/ECCV2022-RIFE/issues/291#issuecomment-1328685348) for **Video Stitching**.

**Image Interpolation**

```bash
python3 inference_img.py --img img0.png img1.png --exp=4
```
(2^4=16X interpolation results)

After that, you can use pngs to generate mp4:
```bash
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -vf "format=yuv420p,colorspace=all=bt709:iall=bt709:fast=1" -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart output/slomo.mp4
```

You can also use pngs to generate gif:
```bash
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -vf "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1" output/slomo.gif
```

### Run in docker (Alternative)

Place the pre-trained models in `train_log/\*.pkl` (as above)

Building the container:
```bash
docker build -t rife -f docker/Dockerfile .
```

Running the container:
```bash
docker run --rm -it -v $PWD:/host rife:latest inference_video --exp=1 --video=untitled.mp4 --output=untitled_rife.mp4
```

```bash
docker run --rm -it -v $PWD:/host rife:latest inference_img --img img0.png img1.png --exp=4
```

Using gpu acceleration (requires proper gpu drivers for docker):
```bash
docker run --rm -it --gpus all -v /dev/dri:/dev/dri -v $PWD:/host rife:latest inference_video --exp=1 --video=untitled.mp4 --output=untitled_rife.mp4
```

## Production Deployment

### Environment Setup

For production deployments, consider these additional configurations:

#### System Requirements
- **GPU**: NVIDIA RTX 2080 Ti or better
- **VRAM**: 4GB minimum, 8GB+ recommended
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: SSD preferred for temporary file operations
- **CPU**: Multi-core processor for video I/O operations

#### Performance Optimization
- Use SSD storage for temp_gradio directory
- Configure appropriate swap space for large videos
- Monitor GPU temperature and throttling
- Set up proper logging and monitoring

#### Security Considerations
- Run containers with non-root user
- Limit file system access with proper volume mounts
- Use secure networking configurations
- Implement rate limiting for web interface
- Regular security updates for base images

### Scaling Strategies

#### Horizontal Scaling
```yaml
# docker-compose.yml for multiple instances
version: '3.8'
services:
  rife-app-1:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./temp_gradio_1:/app/temp_gradio
      - ./train_log:/app/train_log:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  rife-app-2:
    build: .
    ports:
      - "7861:7860"
    volumes:
      - ./temp_gradio_2:/app/temp_gradio
      - ./train_log:/app/train_log:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
```

#### Load Balancing
```nginx
# nginx configuration for load balancing
upstream rife_backend {
    server localhost:7860;
    server localhost:7861;
}

server {
    listen 80;
    location / {
        proxy_pass http://rife_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Monitoring and Logging

#### Health Checks
```dockerfile
# Add to Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1
```

#### Resource Monitoring
```yaml
# docker-compose.yml monitoring
version: '3.8'
services:
  rife-app:
    # ... existing configuration
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### Backup and Recovery

#### Model Backup
```bash
#!/bin/bash
# backup_models.sh
tar -czf models_backup_$(date +%Y%m%d).tar.gz train_log/
aws s3 cp models_backup_$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

#### Data Recovery
```bash
#!/bin/bash
# recover_models.sh
aws s3 cp s3://your-backup-bucket/models_backup_latest.tar.gz .
tar -xzf models_backup_latest.tar.gz
```

### Environment Variables

#### Configuration Options
```bash
# Production environment variables
export CUDA_VISIBLE_DEVICES="0,1"  # GPU selection
export RIFE_MODEL_PATH="/app/train_log"
export RIFE_TEMP_DIR="/app/temp_gradio"
export RIFE_MAX_RESOLUTION="4096"
export RIFE_DEFAULT_SCALE="1.0"
export RIFE_ENABLE_FP16="true"
export GRADIO_SERVER_PORT="7860"
export GRADIO_SERVER_NAME="0.0.0.0"
```

#### Docker Environment File
```bash
# .env file for docker-compose
CUDA_VISIBLE_DEVICES=0,1
RIFE_MODEL_PATH=/app/train_log
RIFE_TEMP_DIR=/app/temp_gradio
RIFE_MAX_RESOLUTION=4096
RIFE_DEFAULT_SCALE=1.0
RIFE_ENABLE_FP16=true
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0
```

## Troubleshooting

### Common Issues

#### GPU Memory Issues
```bash
# Check GPU memory usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce processing scale
python inference_video.py --scale=0.5 --video=input.mp4
```

#### Performance Issues
```bash
# Check system resources
htop
iostat -x 1
nvidia-smi -l 1

# Optimize temporary directory
mkdir -p /tmp/rife_temp
export RIFE_TEMP_DIR=/tmp/rife_temp
```

#### Container Issues
```bash
# Debug container
docker logs rife-app
docker exec -it rife-app /bin/bash

# Rebuild with no cache
docker build --no-cache -t rife-app .
```

### Performance Tuning

#### GPU Optimization
```python
# Add to inference scripts
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

#### Memory Management
```python
# Optimize memory usage
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Integration Examples

### API Integration
```python
# Flask API wrapper
from flask import Flask, request, jsonify
from rife_app.run_interpolation import main_interpolate

app = Flask(__name__)

@app.route('/interpolate', methods=['POST'])
def interpolate_video():
    input_path = request.json['input_path']
    exp_factor = request.json.get('exp', 1)
    
    try:
        output_path = main_interpolate(
            input_video_path=input_path,
            exp=exp_factor,
            use_fp16=True
        )
        return jsonify({'success': True, 'output_path': output_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Batch Processing
```python
# Batch processing script
import os
import glob
from rife_app.run_interpolation import main_interpolate

def batch_interpolate(input_dir, output_dir, exp=1):
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    
    for video_file in video_files:
        try:
            output_path = main_interpolate(
                input_video_path=video_file,
                output_dir_path=output_dir,
                exp=exp,
                use_fp16=True
            )
            print(f"Processed: {video_file} -> {output_path}")
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

# Usage
batch_interpolate('/input/videos', '/output/videos', exp=1)
```

## Summary

This deployment guide provides comprehensive instructions for deploying RIFE in various environments, from local development to production-scale deployments. The Docker-based approach is recommended for most use cases due to its consistency, ease of deployment, and built-in dependency management.

Key deployment strategies:
- **Development**: Local installation with pip
- **Production**: Docker containers with proper resource management
- **Scaling**: Multiple container instances with load balancing
- **Monitoring**: Health checks, logging, and resource monitoring
- **Security**: Proper access controls and security configurations

Choose the deployment method that best fits your infrastructure requirements and operational constraints.