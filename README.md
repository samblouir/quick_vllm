# Quick VLLM

Welcome to **quick VLLM**, a slim Python wrapper that makes talking to a
[vLLM](https://github.com/vllm-project/vllm) or [LM Studio](https://lmstudio.ai/) server easier.

---

## Features

| Feature                           | Description                                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Turn-key**                      | Automatically waits for the vLLM server to boot before sending the first request.                   |
| **Cache**                         | Generated text is cached on disk; re-running the same call with the same settings is instantaneous. |
| **Batching**                      | Built-in multiprocessing lets you hit high throughput with a single function call.                  |
| **Metrics**                       | BLEU score utilities out of the box.                                                                |
| **Configurable**                  | Control cache behaviour and port with CLI flags or environment variables.                           |
| **Multi-host support (NEW)**      | The new `VLLMClient` class lets you spin up *multiple* clients, each with its own host/port, in one script. |

---

## Installation

```bash
pip install --upgrade vllm
pip install git+https://github.com/SamBlouir/quick_vllm.git
````

---

## Starting a vLLM server (example)

```bash
vllm serve \
  meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer meta-llama/Llama-3.2-1B-Instruct \
  --gpu_memory_utilization 0.5 \
  --trust-remote-code \
  --enable-prefix-caching \
  --max-num-seqs 16 \
  --max_model_len 32768 &
```

*See `quick_vllm/vllm_args.py` for a whirlwind of all server flags.*

---

## Python usage

### 1 • Functional API

```python
from quick_vllm import send   # or `import quick_vllm.api as qvapi`

msgs = [
    "Hi there, how are you?",
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Summarise Moby-Dick in two lines."},
    ],
]

responses = send(
    msgs,
    temperature=0.7,
    top_p=0.95,
    n=4,                 # generate 4 responses per prompt
    just_return_text=1,  # -> list[str] instead of list[dict]
    stream_print=True,   # stream-print tokens for the first prompt
)
print(responses)
```

Need asynchronous results?  Use `async_=True` or the
convenience wrapper `send_async`:

```python
handle = send(msgs, async_=True)

# Fetch them all at once
all_responses = handle.get()

# Or grab each one as soon as it's ready
for resp in send_async(msgs):
    print(resp.get())
```

### 2 • Object-oriented client

Need to hit multiple vLLM back-ends in the same programme?  Use the
`VLLMClient` class:

```python
from quick_vllm import VLLMClient

east = VLLMClient(host="10.0.0.11", port=8000)
west = VLLMClient(host="10.0.0.12", port=8001)

print(east.send_message("Ping east", just_return_text=True))
print(west.send_message("Ping west", just_return_text=True))

batch = ["Explain quantum tunnelling in 2 lines.", "Write a haiku about GPU fans."]
print(east.send(batch, just_return_text=True, stream_print=True))        # multiprocess batching still works
```

Want to use LM Studio?
```python
from quick_vllm import VLLMClient

lms_client = quick_vllm.VLLMClient(
    host="192.168.1.169",
    port=1234,
)

example_list = [
    "Hi.",
    "How are you?",
]

scheduled_actions = lms_client.send_async(example_list)

for idx, action in enumerate(scheduled_actions):
    gotten = action.get()
    print(f"Scheduled Action {idx}: {gotten}")
```

Both the functional helpers and the class share the *same* cache directory, so
you can mix and match freely.

---

## Settings quick-reference

| Flag / env var         | Effect                                                        |
| ---------------------- | ------------------------------------------------------------- |
| `--port 8000`          | Point *functional* helpers (`send`, …) at a different port.   |
| `--force_cache_miss 1` | Ignore cached generations *and* overwrite with fresh results. |
| `--disable_cache 1`    | Bypass cache entirely (no load, no save).                     |

> **Note**
> For `VLLMClient` objects just pass `port` / `host` to the constructor. The
> class is completely self-contained.

