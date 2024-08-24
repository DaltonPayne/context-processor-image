# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the model and tokenizer
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16).save_pretrained('/app/model'); \
AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True).save_pretrained('/app/tokenizer');"

# Modify handler.py to load the model from the pre-downloaded path
RUN sed -i "s|AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)|AutoModel.from_pretrained('/app/model', trust_remote_code=True, torch_dtype=torch.float16)|g" handler.py
RUN sed -i "s|AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)|AutoTokenizer.from_pretrained('/app/tokenizer', trust_remote_code=True)|g" handler.py

# Expose port if needed (example: 8080)
# EXPOSE 8080

# Run handler.py when the container launches
CMD ["python", "handler.py"]
