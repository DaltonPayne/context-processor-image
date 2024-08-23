import runpod
import torch
from PIL import Image
import base64
import io
from transformers import AutoModel, AutoTokenizer

# Load pre-downloaded model and tokenizer
model = AutoModel.from_pretrained('/root/.cache/huggingface/hub/models--openbmb--MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')
tokenizer = AutoTokenizer.from_pretrained('/root/.cache/huggingface/hub/models--openbmb--MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

def handler(event):
    try:
        # Get the image data from the event
        image_data = event["input"]["image"]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Set up the question
        question = 'What is in the image?'
        msgs = [{'role': 'user', 'content': question}]
        
        # Generate the response
        with torch.no_grad():
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7,
            )
        
        # Return the result
        return {"status": "success", "description": res}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})