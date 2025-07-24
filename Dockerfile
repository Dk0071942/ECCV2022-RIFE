FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

# Install FFmpeg 7.1 to match local development environment (following LatentSync pattern)
RUN apt-get update && apt-get install -y git wget xz-utils libgl1 libglib2.0-0 libgthread-2.0-0 build-essential python3 python3-pip python3-dev curl --no-install-recommends \
    && wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -O /tmp/ffmpeg.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg* \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code from the build context
COPY . .

# Verify RIFE model files exist (equivalent to LatentSync checkpoints)
RUN echo "RIFE model verification..." && \
    [ -f "/app/train_log/flownet.pkl" ] && \
    [ -f "/app/train_log/RIFE_HDv3.py" ] && \
    [ -f "/app/train_log/IFNet_HDv3.py" ] && \
    echo "RIFE models verified successfully"

# Create necessary directories
RUN mkdir -p temp_gradio

# Expose Gradio port (7860 for RIFE)
EXPOSE 7860

# Set Gradio server configuration
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Authentication Environment Variables
# Set to empty strings to disable authentication (default for local development)
# Override these in deployment to enable authentication
ENV AUTH_USERNAME=""
ENV AUTH_PASSWORD=""

# Add healthcheck with more lenient startup time
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:7860 || exit 1

# Entry point - run the RIFE Gradio app
CMD ["python3", "rife_app/app.py"]