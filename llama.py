

#单卡加载的修改版
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/qiushui/models/Llama-2-7b-chat-hf"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

chat_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,        # 必须 FP16 或 BF16
    low_cpu_mem_usage=True,
    device_map=None                   # ❗单卡必须写 None
).to(device)                          # ❗模型手动放到 GPU

chat_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def get_llama2_chat_response(prompt, max_new_tokens=200):
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = chat_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.00001
    )
    response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#if __name__ == "__main__":
prompt = "Q: What is the capital of India? A:"
response = get_llama2_chat_response(prompt, max_new_tokens=200)
print(response)
