# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional required packages
RUN pip install --no-cache-dir transformers pillow accelerate

# Download and cache the model and tokenizer
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True); AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)"

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run handler.py when the container launches
CMD ["python", "handler.py"]
