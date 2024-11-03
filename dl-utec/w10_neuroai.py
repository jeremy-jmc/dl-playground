"""
Dataset:
    ImageNet 1K

Classificator ViT models:
    - CrossViT
    - DeiT
    - ViT
    - Swin
    - MobileViT
Oracle models:
    - Phi3.5
Adversarial Attacks:
    - PGD
    - FGSM
Independent variables
    - Noise of adversarial attacks
Dependent variables
    - Accuracy of the model
Research question
    - How can we verify that a ViT models is more robust than a another ViT model under adversarial attacks?
    - How can we evaluate the alignment of the oracle model with a ViT model in classificator tasks?
"""
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -----------------------------------------------------------------------------
# Oracle Model: Phi3.5-Vision-Instruct
# -----------------------------------------------------------------------------

from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "microsoft/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
#   _attn_implementation='flash_attention_2'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    num_crops=16
) 

placeholder = ""

img_path_url = ""
img = Image.open(img_path_url)  # requests.get(img_path_url, stream=True).raw
i = 1
placeholder = f"<|image_{i}|>\n"

messages = [
    {"role": "user", "content": placeholder+"Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

generation_args = { 
    "max_new_tokens": 1000, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(**inputs, 
  eos_token_id=processor.tokenizer.eos_token_id, 
  **generation_args
)

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
  skip_special_tokens=True, 
  clean_up_tokenization_spaces=False)[0] 

print(response)

