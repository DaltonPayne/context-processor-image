# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-download the model and tokenizer
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True).save_pretrained('/app/model'); \
    AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True).save_pretrained('/app/tokenizer');"

# Modify handler.py to load the model from the pre-downloaded path
RUN sed -i "s|AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5'|AutoModel.from_pretrained('/app/model'|g" handler.py && \
    sed -i "s|AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5'|AutoTokenizer.from_pretrained('/app/tokenizer'|g" handler.py

# Remove any unnecessary files (adjust as needed)
RUN find /app -type f -name '*.pyc' -delete && \
    find /app -type d -name '__pycache__' -delete

# Run handler.py when the container launches
CMD ["python", "handler.py"]
