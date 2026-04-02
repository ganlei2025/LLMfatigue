import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

model_name = "sshleifer/tiny-gpt2"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型到 GPU（如果可用）
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "Hello, this is a test: "

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
