'''

Collection of cache functions for quick access to data.

Stores content to TMP_DIR in _config.py
This allows you to repeat prompts with the same settings and get the same results.

'''
from quick_vllm import _config
import collections.abc
import hashlib
import json
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    import random as _random

    class _DummyRNG:
        def __init__(self, seed):
            self._rand = _random.Random(seed)

        def random(self):
            return self._rand.random()

        def integers(self, low, high=None):
            if high is None:
                low, high = 0, low
            return self._rand.randint(low, high - 1)

    class _DummyNP:
        class random:
            @staticmethod
            def default_rng(seed=None):
                return _DummyRNG(seed)

    np = _DummyNP()
import os
import pickle
import threading
import time
import traceback



def safe_clear_key(key):
	if isinstance(key, str):
		try:
			os.remove(quick_path_gen(key), )
		except:
			return False
		return True
	else:
		ret_vals = []
		for k in key:
			ret_vals.append(safe_clear_key(k))
		return ret_vals

def cached_func(func=None, key="", flush_cache=False, cache_dir=None):
    """
    A decorator to cache the results of a function.
    The cache key is determined by the function name, the 'key' argument provided
    to the decorator, and the arguments passed to the decorated function at call time.
    """
    if func is None:
        # Allows using @cached_func(key="...", cache_dir="...")
        return lambda f: cached_func(f, key=key, flush_cache=flush_cache, cache_dir=cache_dir)

    def wrapper(*args, **kwargs):
        # Combine the initial key (if any) with call-time args/kwargs for a more specific cache key
        call_specific_key_parts = [str(key)] # Start with the decorator's base key
        
        # Add positional arguments to the key
        for arg in args:
            # Convert arg to a string representation for the key
            # For complex objects, might need a more robust serialization than str()
            call_specific_key_parts.append(str(arg)) 
        
        # Add keyword arguments to the key, sorted for consistency
        for k_kwarg, v_kwarg in sorted(kwargs.items()):
            call_specific_key_parts.append(f"{k_kwarg}={v_kwarg}")
        
        combined_key_str = "_".join(call_specific_key_parts)
        md5_key_part = quick_md5_of_object(combined_key_str) # Use the combined string for MD5
        unique_key = f"{func.__name__}_{md5_key_part}"

        final_cache_dir = cache_dir # Use cache_dir passed to decorator, or None for default

        if not quick_check(unique_key, cache_dir=final_cache_dir, generate_path=True) or flush_cache:
            result = func(*args, **kwargs)
            quick_save(result, out_path=unique_key, cache_dir=final_cache_dir, generate_path=True)
        
        return quick_load(in_path=unique_key, cache_dir=final_cache_dir, generate_path=True)
    
    return wrapper

def cache_path_gen(in_str):
	if '/' in in_str:
		in_str = in_str.rsplit("/", 1)[1]

	if in_str[-6:] == '.cache':
		in_str = in_str[:-6]

	return f"tmp/{in_str}.cache"


def str_to_int(x):
	return int(hashlib.md5(x.encode()).hexdigest(), 16)

def quick_md5_of_object(in_obj):
	# print(f"  in_obj: {in_obj}")
	if isinstance(in_obj, collections.abc.Iterable):
		in_obj = str(in_obj)
		# in_obj = ''.join(list(map(lambda x: str(x), in_obj)))

	in_obj_dumps = pickle.dumps(in_obj)
	# in_obj_dumps = json.dumps(in_obj).encode()

	ret_val = hashlib.md5(in_obj_dumps).hexdigest()
	return str(ret_val)

def quick_hash_of_object(in_obj):
	# print(f"  in_obj: {in_obj}")
	if isinstance(in_obj, collections.abc.Iterable):
		in_obj = str(in_obj)
		# in_obj = ''.join(list(map(lambda x: str(x), in_obj)))

	in_obj_dumps = pickle.dumps(in_obj)
	# in_obj_dumps = json.dumps(in_obj).encode()

	ret_val = hashlib.sha256(in_obj_dumps).hexdigest()
	return f"{ret_val}"

def quick_hash_to_int(in_obj):
	if isinstance(in_obj, collections.abc.Iterable) and not isinstance(in_obj, (str, bytes)):
		# Use JSON for consistent serialization of iterables (e.g., dicts, lists)
		in_obj_dumps = json.dumps(in_obj, sort_keys=True).encode()
	else:
		# Use pickle for other types
		in_obj_dumps = pickle.dumps(in_obj)
	
	# Generate the SHA-256 hash and convert to an integer
	ret_val = hashlib.sha256(in_obj_dumps).hexdigest()
	print(f"  in_obj: {in_obj} -->  ret_val: {ret_val}")
	return int(ret_val, 16)

quick_hash = quick_hash_of_object
hash = quick_hash_of_object
# quick_hash = lambda x: str(quick_hash_to_int(x))
# hash = quick_hash	

def int_hash_of_object(in_obj):
	hash_str = quick_hash_of_object(in_obj)
	hash_int = int(hash_str, 16) % (2**32-1)
	return hash_int

def create_seeded_rng(**x):
	rng_key = int_hash_of_object(x)
	rng = np.random.default_rng(rng_key)
	return rng

def md5_path_gen(in_str):
	if '/' in in_str:
		in_str = in_str.rsplit("/", 1)[1]
	return f"tmp/{in_str}.md5"




def easy_load(in_path, default=None, default_func=None, default_func_args=None):
	if quick_check(in_path):
		return quick_load(in_path)

	if default_func is not None:
		return default_func(*default_func_args)

	return default



def quick_check(in_path, generate_path=True):
	if generate_path:
		in_path = quick_path_gen(in_path)

	return os.path.exists(in_path)


def quick_path_gen(in_obj_name, clean_slashes=True, base_cache_dir=None):
	## Clean off extra slashes
	if clean_slashes:
		in_obj_name = in_obj_name.rsplit("/", 1)[-1]

	base_dir = base_cache_dir if base_cache_dir is not None else _config.TMP_DIR
	
	# Ensure the directory exists
	os.makedirs(base_dir, exist_ok=True)

	return f"{base_dir}/{in_obj_name}.cache"

def quick_check(in_path, generate_path=True, cache_dir=None): # Added cache_dir
	if generate_path:
		in_path = quick_path_gen(in_path, base_cache_dir=cache_dir) # Pass cache_dir

	return os.path.exists(in_path)


def _quick_save(in_obj, out_path=None, generate_path=True, store_in_tmp=True, cache_dir=None): # Added cache_dir
	# if out_path is None and type(in_obj) == str:
	#     out_path = in_obj
	if isinstance(type(in_obj), str) and (not isinstance(type(out_path), str)):
		(in_obj, out_path) = (out_path, in_obj)
		
	if generate_path:
		out_path = quick_path_gen(out_path, base_cache_dir=cache_dir) # Pass cache_dir
	else:
		base_dir = os.path.dirname(out_path)
		if not os.path.exists(base_dir):
			# os.makedirs(base_dir)
			os.system(f"mkdir -p {base_dir}")

	# if store_in_tmp:
	#     split = out_path.split('/', 0)
	#     if split is not None and len(split) > 1:
	#         split = split[0]
	#         if 'tmp' not in split:
	#             out_path = f'/tmp/{out_path}'


	try:
		with open(out_path, 'wb') as f:
			pickle.dump(in_obj, f)
	except:
		os.system(f"mkdir -p {out_path.rsplit('/', 1)[0]}")
		with open(out_path, 'wb') as f:
			pickle.dump(in_obj, f)
	print(f"  Quick saved to {out_path}!")

	# print(f"Quick saved to \"{out_path}\"! (type(in_obj): {type(in_obj)})")
	return out_path

def quick_save(in_obj, out_path=None, generate_path=True, store_in_tmp=True, cache_dir=None): # Added cache_dir
	try:
		assert(isinstance(out_path, str))
		_quick_save(in_obj=in_obj, out_path=out_path, generate_path=generate_path, store_in_tmp=store_in_tmp, cache_dir=cache_dir) # Pass cache_dir
	except:
		print(f"  WARNING:  cache.py  quick_save() exception!  out_path is not a string.", flush=True,)
		_quick_save(in_obj=out_path, out_path=in_obj, generate_path=generate_path, store_in_tmp=store_in_tmp, cache_dir=cache_dir) # Pass cache_dir



def safe_write_to_file(in_obj, out_path, style='json', write_mode='w',):

	# with open(oppa, 'w') as f:
	# 	json.dump(make_json_dumpable(results[task_name]), f)

	in_obj_hash = quick_hash_of_object(in_obj)

	while True:
		if style == 'json':

			try:
				with open(out_path, write_mode) as f:
					json.dump(in_obj, f)
				time.sleep(1)

				# written_hash = quick_hash_of_object(json.load(open(out_path, 'r')))
			except Exception as write_e:
				print(f"  write_e: {write_e}")
				traceback.print_exc()
				time.sleep(5)

			# if (in_obj_hash == written_hash):
			# 	break
			break
			print(f"  WARNING:  cache.py  safe_write_to_file()  Hash mismatch!  Retrying...", flush=True,)
			time.sleep(5)
			continue

		elif style == 'pickle':
			with open(out_path, write_mode) as f:
				pickle.dump(in_obj, f)
			written_hash = quick_hash_of_object(pickle.load(open(out_path, 'rb')))
			if (in_obj_hash == written_hash):
				break
			print(f"  WARNING:  cache.py  safe_write_to_file()  Hash mismatch!  Retrying...", flush=True,)
			time.sleep(5)
			continue
			
		else:
			raise Exception(f"  WARNING:  cache.py  safe_write_to_file()  Unknown style: {style}", flush=True,)

	print(f"  Safely wrote to \"{out_path}\"!", flush=True,)



def quick_safe_save(in_obj, out_path=None, generate_path=True, store_in_tmp=True, ):

	in_obj_hash = quick_hash_of_object(in_obj)

	while True:
		try:
			quick_save(in_obj=in_obj, out_path=out_path, generate_path=generate_path, store_in_tmp=store_in_tmp,)

			loaded = quick_load(out_path)
			loaded_hash = quick_hash_of_object(loaded)

			if in_obj_hash == loaded_hash:
				break

			raise Exception(f"  WARNING:  cache.py  quick_safe_save()  Hash mismatch!  Retrying...", flush=True,)

		except Exception as e:
			print(f"  WARNING:  cache.py  save_safe() exception!  Retrying... e: {e}", flush=True,)
			time.sleep(5)



def async_save(in_obj, out_path=None, generate_path=True, store_in_tmp=True, ):
	thread = threading.Thread(target=quick_save, args=(in_obj, out_path, generate_path, store_in_tmp,))
	try:
		thread.start()
	except Exception as e:
		print(f"  WARNING:  cache.py  async_save() exception!  Running synchronously... e: {e}")
		quick_save(in_obj, out_path, generate_path, store_in_tmp, cache_dir=getattr(threading.current_thread(), 'cache_dir', None)) # Pass cache_dir from thread local or None

	return thread

def quick_load(in_path, generate_path=True, default_value=None, silent=0, cache_dir=None): # Added cache_dir
	if generate_path:
		in_path = quick_path_gen(in_path, base_cache_dir=cache_dir) # Pass cache_dir

	if os.path.exists(in_path):
		# if not silent:
		# 	print(f"  cache.quick_load():  is loading from \"{in_path}\"...", flush=True,)
		try:
			with open(in_path, 'rb') as f:
				return pickle.load(f)
		finally:
			pass
			# if not silent:
			# 	print(f"  cache.quick_load():  Finished loading from \"{in_path}\"!", flush=True,)
	else:
		if default_value is not None:
			return default_value
		raise Exception(f"quick_load({in_path}) failed: File does not exist.")


########

  