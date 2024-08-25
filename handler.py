import runpod
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import base64
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the model and tokenizer
def initialize_model():
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device='cuda')
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
    model.eval()
    return model, tokenizer

model, tokenizer = initialize_model()

def process_image(image_data, question):
    try:
        # Convert base64 to PIL Image
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        
        # Prepare the message
        msgs = [{'role': 'user', 'content': question}]
        
        # Generate response
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7
        )
        
        return res
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return str(e)

def handler(event):
    try:
        # Extract image data and question from the event
        image_data = event.get("input", {}).get("image")
        question = event.get("input", {}).get("question", "What is in the image?")
        
        if not image_data:
            raise ValueError("Image data is required")
        
        # Process the image
        result = process_image(image_data, question)
        
        return {"output": result}
    except Exception as e:
        logging.error(f"Error in handler: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
