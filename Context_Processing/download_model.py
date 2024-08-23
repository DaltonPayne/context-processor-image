import torch
from transformers import AutoModel, AutoTokenizer

def download_model():
    print("Downloading and caching the model...")
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
    
    print("Model and tokenizer downloaded and cached successfully.")

if __name__ == "__main__":
    download_model()