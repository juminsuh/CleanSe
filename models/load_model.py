# This script exists just to load models faster
import functools
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

@functools.lru_cache()
def load_pretrained_model(model_name, torch_dtype=torch.float16): 
    if model_name == 'llama-7b-hf':
        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", device_map = 'auto', cache_dir=None, torch_dtype=torch_dtype) 
    if model_name == 'llama-13b-hf':
        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b", cache_dir=None, torch_dtype=torch_dtype) 
    if model_name == 'Llama-2-7b-hf':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                                    device_map = 'auto', 
                                                    cache_dir=None, 
                                                    torch_dtype=torch_dtype)
    if model_name == 'mistral-7b':
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",  device_map = 'auto', cache_dir=None, torch_dtype=torch_dtype)    
    return model

@functools.lru_cache()
def load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name == 'llama-7b-hf':
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", cache_dir=None, use_fast=use_fast)
    elif model_name == 'llama-13b-hf':
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b", cache_dir=None, use_fast=use_fast)
    elif model_name == 'Llama-2-7b-hf':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                                cache_dir=None, 
                                                use_fast=use_fast)
    elif model_name == 'mistral-7b':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir = None, use_fast = use_fast)

    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
    tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id # set pad token to be same as eos token
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

