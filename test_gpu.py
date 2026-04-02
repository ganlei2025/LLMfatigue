"""
import torch
x = torch.randn(2, 2, device="cuda", requires_grad=True)
y = x * x
z = y.sum()
z.backward()
print("Backward OK, x.grad =", x.grad)
"""
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "sshleifer/tiny-gpt2",
    device_map="auto"
)
print(model.hf_device_map)
