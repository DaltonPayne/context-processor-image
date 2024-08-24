# Use an official Python runtime as a parent image
FROM python:3.10-slim AS builder

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install build dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Pre-download the model and tokenizer
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True).save_pretrained('/app/model'); \
    AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True).save_pretrained('/app/tokenizer');"

# Modify handler.py to load the model from the pre-downloaded path
RUN sed -i "s|AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5'|AutoModel.from_pretrained('/app/model'|g" handler.py && \
    sed -i "s|AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5'|AutoTokenizer.from_pretrained('/app/tokenizer'|g" handler.py

# Start a new stage with a clean image
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files from the builder stage
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Run handler.py when the container launches
CMD ["python", "handler.py"]
