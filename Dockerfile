FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install system dependencies (following LatentSync pattern)
RUN apt-get update && apt-get install -y git ffmpeg libgl1 build-essential python3 python3-pip python3-dev curl --no-install-recommends \
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

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# Entry point - run the RIFE Gradio app
CMD ["python3", "rife_app/app.py"]