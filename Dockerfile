FROM nvidia/cuda:12.6-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the RIFE app code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p temp_gradio

# Expose Gradio port
EXPOSE 7860

# Set Gradio server configuration
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Set Python path to include the RIFE directory
ENV PYTHONPATH="/app:$PYTHONPATH"

# Entry point - run the RIFE Gradio app
CMD ["python3", "rife_app/app.py"]