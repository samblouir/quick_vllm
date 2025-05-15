# Quick VLLM
Welcome to quick VLLM, a simple Python script that enables easy asynchronous and batched usage of VLLM in Python!
It is designed to help you do fast batch inference with VLLM.
## Features
- **Turn-key**: Just run the script and it will automatically load the model and start generating text.
- **Cache**: The generated text will be cached to disk, so you can reload generations later.
- **Batching**: The script will automatically batch the input text and generate text in parallel.
- **Metrics**: Currently, there is limited BLEU score support.
- **Configurable**: You can easily enable or disable the cache, as well as force cache misses.

## Usage
### Start VLLM
```bash
vllm serve \
	meta-llama/Llama-3.2-1B-Instruct \
	--tokenizer meta-llama/Llama-3.2-1B-Instruct \ 
	--gpu_memory_utilization 0.5 \
	--trust-remote-code  \
	--kv-cache-dtype auto \
	--enable-prefix-caching \
	--max-num-seqs 16 \
	--max_model_len 32768

# tokenizer: Useful if a model's config is bugged and doesn't have a tokenizer defined
# kv-cache-dtype: Set to fp8 to save VRAM, will lower quality
# enable-prefix-caching: Enable prefix caching for faster generation
# max-num-seqs: The number of sequences to generate in parallel. Increase this for greater parallel generation speeds (i.e.: 512+ for an A100 80GB). 
# - Note: Llama 3.2 1B uses ~11GB of VRAM with `max-num-seqs 16`.
# max_model_len: The maximum sequence length: shared between the context and the generated text.
```

### Start Python code
#### Example usage of quick_vllm
```bash
python quick_vllm/example_usage.py 
```

### Cache Control Settings
The cache will automatically save and load generated text.
#### To disable loading from the cache:
```bash
python quick_vllm/example_usage.py --force_cache_miss 1
```


#### To disable saving to the cache:
```bash
python quick_vllm/example_usage.py --disable_cache 1
```
