import openai
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import time

import transformers
# Use a pipeline as a high-level helper
from transformers import pipeline

from transformers import LlamaForCausalLM, LlamaTokenizer
import os


def call_openai_server_single_prompt(
    prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8, seed=1
):
  """The function to call OpenAI server with an input string."""
  try:
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_decode_steps,
        messages=[
            {"role": "user", "content": prompt},
        ],
        seed=seed
    )
    return completion.choices[0].message.content

  except openai.error.Timeout as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.RateLimitError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"API error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIConnectionError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"API connection error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.ServiceUnavailableError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"Service unavailable. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except OSError as e:
    retry_time = 60  # Adjust the retry time as needed
    print(
        f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
    )
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )


def call_openai_server_func(
    inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8, seed=1,
    model_pretrain=None,
    tokenizer=None
):
  """The function to call OpenAI server with a list of input strings."""
  if isinstance(inputs, str):
    inputs = [inputs]
  outputs = []
  if 'deepseek' in model:
    output = call_llama_single_prompt(
        inputs[0],
        max_decode_steps=max_decode_steps,
        temperature=temperature,
        seed=seed,
        model_pretrain=model_pretrain,
        tokenizer=tokenizer
    )
    outputs.append(output)
  else:

    for input_str in inputs:

      output = call_openai_server_single_prompt(
          input_str,
          model=model,
          max_decode_steps=max_decode_steps,
          temperature=temperature,
          seed=seed
      )
      outputs.append(output)
  return outputs


def load_model(model_name):

    model_dir = "/home/qiushui/models/deepseek-math-7b-instruct"

    # 原来写的是 "Loading Llama-2 model from"，这里稍微改一下避免误导
    print("Loading model from:", model_dir)

    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,   # 单卡 GPU 推荐半精度
        device_map="auto"            # 自动把模型放到 GPU 上
    )

    # 关键修改：deepseek-math-7b-base 用 AutoTokenizer（fast），不要用 LlamaTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
    )

    # 如果后面代码里会用到 pad_token，给它一个默认值（否则有时会 warning / 报错）
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def call_llama_single_prompt(
    input_str,
    max_decode_steps,
    temperature,
    seed,
    model_pretrain,
    tokenizer
):

    print('pipeline building')
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_pretrain,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
    return_full_text = False,
        # truncation=True
    )
    print("inferences")

    # ===== 关键修改：防止 temperature = 0.0 =====
    if temperature is None or temperature <= 0:
        print(f"[警告] 收到 temperature={temperature}，自动改为 0.7 以避免 transformers 报错")
        temperature = 0.7
    # =========================================

    sequences = pipeline(
        input_str,
        do_sample=True,          # 继续用采样
        top_k=20,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_decode_steps,
        temperature=temperature  # 这里已经保证 > 0
    )

    generations = sequences[0]['generated_text'].split('\n')
    generations_del = [gen for gen in generations if len(gen) > 0]
    return "\n".join(generations_del)
