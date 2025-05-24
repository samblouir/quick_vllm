from openai import OpenAI
import multiprocessing as mp
from multiprocessing import pool as mp_pool
import os
import traceback
import base64
import time
import datetime
import copy
from quick_vllm import cache
from quick_vllm.utils import arg_dict


mdl = arg_dict.get("mdl", "UNKNOWN_MODEL")
# Defaults to port 8000
port = str(int(arg_dict.get("port", 8000)))

default_sampling_parameters = {
    "min_p": 0.05,
    "top_p": 0.95,
    "temperature": 0.7,
    "top_k": 40,
    "max_tokens": int(arg_dict.get("max_tokens", 4096)),
}


def get_client():
    client = OpenAI(base_url=f"http://localhost:{port}/v1/", api_key="vllm")
    return client
pass


def get_mdl(client=None):
    if client is None:
        client = get_client()
        
    while True:
        try:
            models = client.models.list()
            break
        except Exception as e:
            print(f"*" * 60,)
            print(f"  {datetime.datetime.now()},  {__file__}  get_mdl():  Exception e: {e}")
            time.sleep(1)
            
    for model in models:
        return model.id
    
    raise Exception(f"  {datetime.datetime.now()}  Error getting models from VLLM!")
pass



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string
pass


def print_chat(
    responses,
    clear_thinking=False,
    only_thinking=False,
    only_box=False,
):
    to_print = [f"*" * 60]
    for responses_idx, (key, value) in enumerate(responses.items()):
        if isinstance(value, dict) and "text" in value:
            value = value["text"]
        else:
            value = "".join(value)
        
        if not len(value):
            continue

        if only_box:
            if "<box>" in value and "</box>" in value:
                value = value.rsplit("<box>", 1)[-1].rsplit("</box>", 1)[0]
            else:
                value = ""
        else:
            if only_thinking:
                if "</think>" in value:
                    value = value.split("</think>", 1)[0] + "</think>"
            if clear_thinking:
                value = value.rsplit("</think>", 1)[-1]

        to_print.extend(
            [
                f"\n",
                f"-" * 40,
                f"[[{key}]]",
                f"{value}",
                f"-" * 40,
                f"\n",
            ]
        )

    to_print.append(f"*" * 60)
    to_print = "\n".join(to_print)
    print(f"\n" * 4, to_print, sep="")
pass

def remove_invalid_messages(messages):
    return [d for d in messages if d.get("content", "") is not None]
pass

def _run_message(messages, cache_dir=None, **kwargs): # Added cache_dir
    timeout_retry_delay_interval = 5
    kwargs = copy.deepcopy(kwargs)
    stream = kwargs.pop("stream", True)
    # silent = kwargs.pop("silent", False) # set to False to print and stream responses from the model as they are generated
    silent = kwargs.pop("silent", True) # set to False to print and stream responses from the model as they are generated
    client = kwargs.pop("client", None)
    force_cache_miss = kwargs.pop("force_cache_miss", False)
    messages = remove_invalid_messages(messages)

    allowed_body_kwargs = [
        "min_p",
        "top_p",
        "temperature",
        "top_k",
        "max_tokens",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "n",
    ]
    body_kwargs = {k: v for k, v in kwargs.items() if k in allowed_body_kwargs}
    body_kwargs = {k: v for k, v in body_kwargs.items() if v is not None}


    settings = dict(
        model=kwargs.get("model", mdl),
        messages=messages,
        stream=stream,
        extra_body={
            
            **default_sampling_parameters,
            **body_kwargs,
        },
    )
    settings.update(kwargs)
    settings = dict(sorted(settings.items()))
    settings["extra_body"] = dict(sorted(settings["extra_body"].items()))

    settings_keys_to_keep = [
        "model",
        "messages",
        "stream",
        "extra_body",
    ]
    settings = {k: v for k, v in settings.items() if k in settings_keys_to_keep and k not in allowed_body_kwargs}
        
    cache_settings = {k:v for k, v in settings.items() if k not in ["stream",]}
    # for cache_settings_idx,(key,value) in enumerate(cache_settings.items()):
    #     print(f"  cache_settings[{key}]: {value}")
        
    out_path = cache.quick_hash(cache_settings)

    try:
        if force_cache_miss:
            raise Exception("Force cache miss")


        if int(arg_dict.get("force_cache_miss", 0)):
            # Global force cache miss from when the Python script is run with force_cache_miss=1
            raise Exception("Force cache miss")

        if int(arg_dict.get("disable_cache", 0)):
            # Global disable cache from when the Python script is run with disable_cache=1
            raise Exception("Force cache miss")
        
        output = cache.quick_load(out_path, cache_dir=cache_dir) # Pass cache_dir
        
        if not isinstance(output, list):
            # Error detected
            raise Exception("Force cache miss")
        
        if len(output) == 0:
            # Error detected
            raise Exception("Force cache miss")
        
        return output
    
    except:
        if client is None:
            client = get_client()

        while True:
            try:
                completion = client.chat.completions.create(**settings, timeout=999_999)
                break
            except Exception as e:
                print(f"  {__file__} _run_message():  e: {e}")
                time.sleep(timeout_retry_delay_interval)
                timeout_retry_delay_interval += 5
                timeout_retry_delay_interval = min(timeout_retry_delay_interval, 30)


        responses = {}

        if settings.get("stream", True):
            while True:
                try:
                    for chunk in completion:
                        if True:
                            choice = chunk.choices[0]
                            index = choice.index
                            content = choice.delta.content
                            if content is not None:
                                responses[index] = responses.get(index, []) + [content]

                            if not silent:
                                print_chat(responses)

                            continue
                    break

                except Exception as e:
                    print(f" *** " * 4)
                    print(f"  api.py  Broke on: {e}")
                    traceback.print_exc()
                    print(f" *** " * 4)
                    time.sleep(1)

            responses = {k: "".join(v) for k, v in responses.items()}
            responses = [v for k, v in responses.items()]
            responses = list(set(responses)) # Remove duplicates
        else:
            responses = completion.choices

            # for responses_idx, (_responses) in enumerate(responses):
            # 	print(f"  responses[{responses_idx}]: {_responses}")

            responses = {
                idx: {
                    **settings,
                    "finish_reason": completion.choices[idx].finish_reason,
                    "index": completion.choices[idx].index,
                    "stop_reason": completion.choices[idx].stop_reason,
                    "text": completion.choices[idx].message.content,
                }
                for idx in range(len(completion.choices))
            }

            if not int(arg_dict.get("disable_cache", 0)):
                cache.quick_save(responses, out_path, cache_dir=cache_dir) # Pass cache_dir

            return responses
        
        settings_without_stream = {k: v for k, v in settings.items() if k not in ["stream",]}
        responses = {
            idx: {
                "text": responses[idx],
                "settings": settings_without_stream,
            }
            for idx in range(len(responses))
        }

        if not silent:
            os.system("clear")
            print_chat(
                responses,
                only_thinking=True,
            )
            print(f"#" * 60,)
            print(f"#" * 60,)
            print_chat(responses, clear_thinking=True)

        responses = [v for v in responses.values()]
        [r.pop("stream", None) for r in responses]

        if int(arg_dict.get("use_cache", 0)):
            cache.quick_save(responses, out_path, cache_dir=cache_dir) # Pass cache_dir

        return responses
pass



def send_message(
    msg,
    system_prompt=None,
    cache_dir=None, # Added cache_dir
    **kwargs,
):
    if isinstance(msg, str):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg},
        ]
    else:
        msgs = msg

    response = _run_message(
        msgs,
        model=get_mdl(), # Gets the mdl each generation, in case the loaded model has changed in VLLM
        cache_dir=cache_dir, # Pass cache_dir
        **kwargs,
    )
    return response
pass


def send_message_no_cache(msg, **kwargs):
    return send_message(
        msg,
        force_cache_miss=1,
        **{
            **kwargs,
        },
    )
pass



def _batch_send_message_wrapper(d):
    msg = d.get("msg", None)
    kwargs = d.get("kwargs", {})
    force_cache_miss = kwargs.get("force_cache_miss", 0)
    just_return_text = kwargs.get("just_return_text", 0)
    cache_dir = kwargs.pop("cache_dir", None) # Extract cache_dir, remove from kwargs

    if force_cache_miss:
        _send_message = send_message_no_cache
    else:
        _send_message = send_message
    
    # Pass cache_dir to _send_message. send_message_no_cache will also accept cache_dir but ignore it internally if needed.
    response = _send_message(msg, cache_dir=cache_dir, **kwargs,) 
    
    if just_return_text:
        return [r["text"] for r in response]
    
    return response
pass



class _AsyncSendItem:
    """Wrapper around :class:`multiprocessing.pool.AsyncResult`."""

    def __init__(self, async_result, parent):
        self._async_result = async_result
        self._parent = parent
        self._value = None
        self._retrieved = False

    def get(self, *args, **kwargs):
        if not self._retrieved:
            self._value = self._async_result.get(*args, **kwargs)
            self._retrieved = True
            self._parent._item_done()
        return self._value


class _AsyncSendResult:
    """Handle for asynchronous :func:`send` calls."""

    def __init__(self, async_results, pool):
        self._pool = pool
        self._items = [_AsyncSendItem(r, self) for r in async_results]
        self._pending = len(self._items)

    # --------------------------------------------------------------
    def _item_done(self) -> None:
        self._pending -= 1
        if self._pending == 0:
            self._pool.join()

    # --------------------------------------------------------------
    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def get(self, *args, **kwargs):
        return [item.get(*args, **kwargs) for item in self._items]


def send(
    msgs,
    max_pool_size=mp.cpu_count(),
    cache_dir=None,  # Added cache_dir
    *,
    async_=False,
    stream_print=False,
    **kwargs,
):
    if isinstance(msgs, str):
        msgs = [msgs]

    max_pool_size = min(max_pool_size, len(msgs))

    # Use threads when asynchronous streaming is requested so that
    # tokens printed by worker tasks appear in the main console.
    if async_ and stream_print:
        pool = mp_pool.ThreadPool(processes=max_pool_size)
    else:
        pool = mp.Pool(processes=max_pool_size)

    # Include cache_dir in the kwargs for each message
    current_call_kwargs = {**kwargs, "cache_dir": cache_dir}

    msgs_with_kwargs = []
    for idx, msg in enumerate(msgs):
        item_kwargs = dict(current_call_kwargs)
        if stream_print and "silent" not in item_kwargs:
            item_kwargs["silent"] = idx != 0
        msgs_with_kwargs.append(dict(msg=msg, kwargs=item_kwargs))

    if async_:
        async_results = [
            pool.apply_async(_batch_send_message_wrapper, (d,))
            for d in msgs_with_kwargs
        ]
        pool.close()
        return _AsyncSendResult(async_results, pool)

    responses = pool.map(_batch_send_message_wrapper, msgs_with_kwargs)
    pool.close()
    pool.join()
    return responses
pass


def send_async(msgs, max_pool_size=mp.cpu_count(), cache_dir=None, stream_print=False, **kwargs):
    """Convenience wrapper around :func:`send` with ``async_=True``."""
    return send(
        msgs,
        max_pool_size=max_pool_size,
        cache_dir=cache_dir,
        async_=True,
        stream_print=stream_print,
        **kwargs,
    )

