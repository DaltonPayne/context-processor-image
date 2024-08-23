# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler code and model download script
COPY handler.py .
COPY download_model.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Download and cache the model during the build process
RUN python download_model.py

# Run the handler
CMD ["python", "handler.py"]