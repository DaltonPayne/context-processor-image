import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import base64
from io import BytesIO
import logging
import traceback
import os
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to store the model and tokenizer
model = None
tokenizer = None

def log_system_info():
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU model: {torch.cuda.get_device_name(0)}")
    logger.info(f"CPU count: {os.cpu_count()}")
    logger.info(f"Total memory: {psutil.virtual_memory().total / (1024 * 1024 * 1024):.2f} GB")

def resize_and_encode_image(image_data, max_size=(1344, 1344)):
    try:
        logger.info("Starting image processing")
        # Decode base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Log original image size
        logger.info(f"Original image size: {image.size}")
        
        # Resize image if it's larger than max_size
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)
            logger.info(f"Image resized to: {image.size}")
        else:
            logger.info("Image size within limits, no resize needed")
        
        # Convert image to RGB mode
        image = image.convert('RGB')
        logger.info("Image converted to RGB")
        
        return image
    except Exception as e:
        logger.error(f"Error in resize_and_encode_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def load_model():
    global model, tokenizer
    try:
        if model is None or tokenizer is None:
            logger.info("Starting model and tokenizer loading")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Contents of /app: {os.listdir('/app')}")
            
            logger.info("Loading model...")
            model = AutoModel.from_pretrained('/app/model', trust_remote_code=True, torch_dtype=torch.float16)
            logger.info("Model loaded successfully")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Moving model to device: {device}")
            model = model.to(device=device)
            
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('/app/tokenizer', trust_remote_code=True)
            logger.info("Tokenizer loaded successfully")
            
            model.eval()
            logger.info("Model set to evaluation mode")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def log_memory_usage():
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    logger.info(f"CPU memory usage: {psutil.virtual_memory().percent}%")

def process_messages(messages):
    logger.info("Processing messages")
    processed_msgs = []
    for msg in messages:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            processed_msgs.append(msg)
        elif isinstance(msg, str):
            processed_msgs.append({'role': 'user', 'content': msg})
        else:
            logger.warning(f"Skipping invalid message format: {msg}")
    logger.info(f"Processed {len(processed_msgs)} messages")
    return processed_msgs

def handler(event):
    global model, tokenizer
    try:
        logger.info("Handler function started")
        log_system_info()
        
        # Load the model if it's not already loaded
        model, tokenizer = load_model()
        
        # Extract input from the event
        input_data = event.get('input', {})
        image_data = input_data.get('image')
        messages = input_data.get('messages', [])
        sampling = input_data.get('sampling', True)
        temperature = input_data.get('temperature', 0.7)
        stream = input_data.get('stream', False)
        
        logger.info(f"Input parameters - Messages count: {len(messages)}, Sampling: {sampling}, Temperature: {temperature}, Stream: {stream}")
        
        # Process messages
        processed_msgs = process_messages(messages)
        
        # Process image if provided
        if image_data:
            try:
                image = resize_and_encode_image(image_data)
                logger.info(f"Processed image size: {image.size}")
            except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}")
        else:
            logger.info("No image data provided, proceeding with text-only input")
            image = None
        
        # Generate the response
        logger.info("Starting model inference")
        log_memory_usage()
        
        if stream:
            logger.info("Using streaming mode for response generation")
            res = model.chat(
                image=image,
                msgs=processed_msgs,
                tokenizer=tokenizer,
                sampling=sampling,
                temperature=temperature,
                stream=True
            )
            generated_text = ""
            for new_text in res:
                generated_text += new_text
            logger.info(f"Generated text length: {len(generated_text)}")
        else:
            logger.info("Using non-streaming mode for response generation")
            res = model.chat(
                image=image,
                msgs=processed_msgs,
                tokenizer=tokenizer,
                sampling=sampling,
                temperature=temperature
            )
            logger.info(f"Generated text length: {len(res)}")
        
        log_memory_usage()
        logger.info("Handler function completed successfully")
        return {"generated_text": generated_text if stream else res}
    
    except ValueError as ve:
        logger.error(f"Value error in handler: {str(ve)}")
        return {"error": str(ve)}
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory error")
        log_memory_usage()
        return {"error": "GPU out of memory. Please try again later or with a smaller image."}
    except Exception as e:
        logger.error(f"Unexpected error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        log_memory_usage()
        return {"error": "An unexpected error occurred. Please try again later."}

# Log system info at startup
log_system_info()
