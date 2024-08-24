import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import base64
from io import BytesIO
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store the model and tokenizer
model = None
tokenizer = None

def resize_and_encode_image(image_data, max_size=(1344, 1344)):
    try:
        # Decode base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Log original image size
        logger.info(f"Original image size: {image.size}")
        
        # Resize image if it's larger than max_size
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)
            logger.info(f"Image resized to: {image.size}")
        
        # Convert image to RGB mode
        image = image.convert('RGB')
        
        # Encode image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in resize_and_encode_image: {str(e)}")
        raise

def load_model():
    global model, tokenizer
    try:
        if model is None or tokenizer is None:
            logger.info("Loading model and tokenizer...")
            model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
            model = model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
            model.eval()
            logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except ImportError as e:
        logger.error(f"Required package missing: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def handler(event):
    global model, tokenizer
    try:
        # Load the model if it's not already loaded
        model, tokenizer = load_model()

        # Extract input from the event
        input_data = event.get('input', {})
        image_data = input_data.get('image')
        question = input_data.get('question', 'What is in the image?')
        sampling = input_data.get('sampling', True)
        temperature = input_data.get('temperature', 0.7)
        stream = input_data.get('stream', False)

        if not image_data:
            raise ValueError("No image data provided.")

        # Resize and re-encode the image
        try:
            resized_image_data = resize_and_encode_image(image_data)
        except Exception as e:
            raise ValueError(f"Error resizing image: {str(e)}")

        # Decode and open the resized image
        try:
            image = Image.open(BytesIO(base64.b64decode(resized_image_data))).convert('RGB')
            logger.info(f"Processed image size: {image.size}")
        except Exception as e:
            raise ValueError(f"Invalid image data after resizing: {str(e)}")

        # Prepare the messages
        msgs = [{'role': 'user', 'content': question}]

        # Generate the response
        if stream:
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=sampling,
                temperature=temperature,
                stream=True
            )
            generated_text = ""
            for new_text in res:
                generated_text += new_text
            return {"generated_text": generated_text}
        else:
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=sampling,
                temperature=temperature
            )
            return {"generated_text": res}

    except ValueError as ve:
        logger.error(f"Value error in handler: {str(ve)}")
        return {"error": str(ve)}
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory error")
        return {"error": "GPU out of memory. Please try again later or with a smaller image."}
    except ImportError as ie:
        logger.error(f"Import error: {str(ie)}")
        return {"error": "Required package is missing. Please install the necessary dependencies."}
    except Exception as e:
        logger.error(f"Unexpected error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": "An unexpected error occurred. Please try again later."}
