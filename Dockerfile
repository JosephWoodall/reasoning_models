# Use NVIDIA's latest PyTorch container with CUDA support
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be needed
RUN pip install --no-cache-dir \
    PyYAML \
    jupyter \
    tensorboard \
    matplotlib \
    seaborn

# Create output directories
RUN mkdir -p /workspace/output/checkpoints \
    && mkdir -p /workspace/output/logs \
    && mkdir -p /workspace/output/samples

# Copy project files (this will be overridden by volume mount in docker-compose)
COPY . /workspace/

# Set default command
CMD ["python", "src/training_pipeline/train_model.py"]
