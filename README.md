# Quick VLLM
Welcome to quick VLLM, a simple Python script that enables easy asynchronous and batched usage of VLLM in Python!
It is designed to help you do fast batch inference with VLLM.
## Features
- **Turn-key**: Just run the script and it will automatically wait for the VLLM server to boot.
- **Cache**: The generated text will be cached to disk, so you can reload generations later.
- **Batching**: The script will automatically batch the input text and generate text in parallel.
- **Metrics**: Currently, there is limited BLEU score support.
- **Configurable**: You can easily enable or disable the cache, force cache misses to update the cache, and the VLLM port.

## Installation
```bash
pip install git+https://github.com/SamBlouir/quick_vllm.git
```

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
	--max_model_len 32768 & # Include the ampersand to run this in the background. Later, type `fg` to take control and ctrl-c to stop the server.

# tokenizer: Useful if a model's config is bugged and doesn't have a tokenizer defined
# kv-cache-dtype: Set to fp8 to save VRAM, will lower quality
# enable-prefix-caching: Enable prefix caching for faster generation
# max-num-seqs: The number of sequences to generate in parallel. Increase this for greater parallel generation speeds (i.e.: 512+ for an A100 80GB). 
# - Note: Llama 3.2 1B uses ~11GB of VRAM with `max-num-seqs 16`.
# max_model_len: The maximum sequence length: shared between the context and the generated text.
```

### Python Usage

```python
import quick_vllm.api as qvapi

# Example usage
msgs = [
	"Hi there, how are you?",
	"The weather is so nice today.",
	"I can't wait to see you.",
]

## OR

msgs = [
	[
		{"role":"system", "content": "You are a helpful assistant."},
		{"role":"user", "content": "Hi there, how are you?"},
	],

	[
		{"role":"system", "content": "You are an assistant who enjoys the sun."},
		{"role":"user", "content": "The weather is so nice today."},
	],
]


# Set to 1 to just returns a simple list of text responses
# Otherwise, it returns a list of dictionaries with the full response and settings used
just_return_text = 0

message_responses = qvapi.send(
	msgs,
	just_return_text=just_return_text,
	temperature=0.7,
	top_p=0.95,
	min_p=0.0,
	n=4, # Generate 4 responses per message
)
```

By default, this will return a nested list of dictionaries with the generated text and the settings used.
```bash
[
  [
    {
      "text": str, # The generated text from the model
      "settings": {
        "extra_body": {
          "max_tokens": int,
          "min_p": float,
          "n": int,
          "temperature": float,
          "top_k": int,
          "top_p": float
        },
        "messages": [
          {
            "role": str
            "content": str
          }
        ],
        "model": str
      }
    },

	[... 3 more entries in this list]
  ],

  [ ... Another list of 4 dicts for our second prompt will be here ]
]

```

#### Example usage of quick_vllm
```bash
python quick_vllm/example_usage.py 
```

### Settings
#### VLLM Port
The default port for VLLM is 8080. You can change this by setting the `VLLM_PORT` environment variable or **by specifying a different port when starting the script**.
```bash
python quick_vllm/example_usage.py --port 9999
```

#### Cache Control Settings
The cache will automatically save and load generated text.
##### To disable loading from the cache:
```bash
python quick_vllm/example_usage.py --force_cache_miss 1
```


##### To disable saving to the cache:
```bash
python quick_vllm/example_usage.py --disable_cache 1
```

### TODO: Finish adding async.

## Example Output
```markdown
Message Response 1:
  input_messages: {
  "extra_body": {
    "max_tokens": 4096,
    "min_p": 0.0,
    "n": 4,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95
  },
  "messages": [
    {
      "role": "system",
      "content": "You are an assistant who enjoys the sun."
    },
    {
      "role": "user",
      "content": "The weather is so nice today."
    }
  ],
  "model": "meta-llama/Llama-3.2-1B-Instruct"
}
************************
  message_responses[0]:
'''
Isn't it though? I love days like this when the sun is shining brightly. It's perfect for going for a walk, having a picnic, or just spending time outside. I'm glad you're enjoying it too! How about I suggest a plan for a fun outdoor activity today?
'''
************************
  message_responses[1]:
'''
That's wonderful! A beautiful day like today is just what we need. There's something so uplifting about the warmth of the sun on our skin. I'm glad you're enjoying it too. Maybe we can spend some time outside, get some sun and relax?
'''
************************
  message_responses[2]:
'''
Isn't it though? I just love days like this where the sun shines bright and everything feels so warm and cozy. There's nothing like basking in the warmth and taking a walk outside to get some fresh air and enjoy the beauty of nature. Do you have any plans for the day?
'''
************************
  message_responses[3]:
'''
Isn't it though? The sunshine is just perfect. There's something so uplifting about spending time outside on a beautiful day. I'm actually looking forward to taking a walk in the park later.
'''
************************************************************
```