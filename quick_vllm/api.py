from openai import OpenAI
import multiprocessing as mp
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

def _run_message(messages, **kwargs):
    timeout_retry_delay_interval = 5
    kwargs = copy.deepcopy(kwargs)
    cache_dir = kwargs.pop("cache_dir", None)
    if cache_dir is not None:
        _config.set_TMP_DIR(cache_dir)
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
        
        output = cache.quick_load(out_path, cache_dir=cache_dir)
        
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
                cache.quick_save(responses, out_path, cache_dir=cache_dir)

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
            cache.quick_save(responses, out_path, cache_dir=cache_dir)

        return responses
pass



def send_message(
    msg,
    system_prompt=None,
    cache_dir: str | None = None,
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
        model=get_mdl(),  # Gets the mdl each generation, in case the loaded model has changed in VLLM
        cache_dir=cache_dir,
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

    if force_cache_miss:
        _send_message = send_message_no_cache
    else:
        _send_message = send_message

    response = _send_message(msg, **kwargs,)
    
    if just_return_text:
        return [r["text"] for r in response]
    
    return response
pass



def send(
    msgs,
    max_pool_size=mp.cpu_count(),
    **kwargs,
):
    if isinstance(msgs, str):
        msgs = [msgs]

    max_pool_size = min(max_pool_size, len(msgs))
    pool = mp.Pool(processes=max_pool_size)

    # pool = mp.pool.ThreadPool(processes=max_pool_size)

        
    msgs_with_kwargs = [dict(msg=msg, kwargs=kwargs) for msg in msgs]
    responses = pool.map(_batch_send_message_wrapper, msgs_with_kwargs)

    pool.close()
    pool.join()

    return responses
pass

