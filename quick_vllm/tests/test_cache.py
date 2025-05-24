import unittest
import os
import shutil
import tempfile
import pickle
import multiprocessing as mp
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout

from quick_vllm import cache
from quick_vllm import api
from quick_vllm import _config
from quick_vllm import vllm_client

class TestCacheFunctionality(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing cache
        self.temp_dir_root = tempfile.mkdtemp()
        # Define a default test cache dir that would be normally _config.TMP_DIR
        self.default_test_cache_dir = os.path.join(self.temp_dir_root, "default_cache")
        os.makedirs(self.default_test_cache_dir, exist_ok=True)
        
        # Mock _config.TMP_DIR to use our temp default_test_cache_dir
        self.mock_config_tmp_dir = patch.object(_config, 'TMP_DIR', self.default_test_cache_dir)
        self.mock_config_tmp_dir.start()

    def tearDown(self):
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir_root)
        self.mock_config_tmp_dir.stop()

    def test_quick_path_gen_default_dir(self):
        obj_name = "test_object"
        expected_path = os.path.join(self.default_test_cache_dir, f"{obj_name}.cache")
        # _config.TMP_DIR is mocked to self.default_test_cache_dir
        generated_path = cache.quick_path_gen(obj_name)
        self.assertEqual(generated_path, expected_path)
        self.assertTrue(os.path.exists(self.default_test_cache_dir)) # quick_path_gen creates the dir

    def test_quick_path_gen_custom_dir(self):
        obj_name = "test_object_custom"
        custom_dir = os.path.join(self.temp_dir_root, "custom_cache")
        
        expected_path = os.path.join(custom_dir, f"{obj_name}.cache")
        generated_path = cache.quick_path_gen(obj_name, base_cache_dir=custom_dir)
        
        self.assertEqual(generated_path, expected_path)
        self.assertTrue(os.path.exists(custom_dir)) # Check if custom_dir was created

    def test_save_load_default_dir(self):
        obj_name = "my_data_default"
        data = {"key": "value", "number": 123}
        
        # _config.TMP_DIR is mocked
        cache.quick_save(data, obj_name)
        loaded_data = cache.quick_load(obj_name)
        
        self.assertEqual(data, loaded_data)
        expected_file_path = os.path.join(self.default_test_cache_dir, f"{obj_name}.cache")
        self.assertTrue(os.path.exists(expected_file_path))

    def test_save_load_custom_dir(self):
        obj_name = "my_data_custom"
        data = {"key": "custom_value", "number": 456}
        custom_cache_path = os.path.join(self.temp_dir_root, "user_specific_cache")
        # No need to os.makedirs, quick_save should handle it via quick_path_gen

        cache.quick_save(data, obj_name, cache_dir=custom_cache_path)
        loaded_data = cache.quick_load(obj_name, cache_dir=custom_cache_path)

        self.assertEqual(data, loaded_data)
        
        expected_file_path = os.path.join(custom_cache_path, f"{obj_name}.cache")
        self.assertTrue(os.path.exists(expected_file_path))
        
        default_dir_file_path = os.path.join(self.default_test_cache_dir, f"{obj_name}.cache")
        self.assertFalse(os.path.exists(default_dir_file_path))

    def test_load_non_existent_custom_dir(self):
        obj_name = "non_existent_data"
        custom_cache_path = os.path.join(self.temp_dir_root, "another_custom_cache")
        # Ensure the directory exists for the test to be valid, but the file shouldn't
        os.makedirs(custom_cache_path, exist_ok=True)

        with self.assertRaises(Exception) as context: # Assuming quick_load raises a generic Exception
            cache.quick_load(obj_name, cache_dir=custom_cache_path)
        self.assertTrue(f"quick_load({os.path.join(custom_cache_path, obj_name + '.cache')}) failed: File does not exist." in str(context.exception))
        
    # For testing cached_func, we need a way to count function calls
    # We can use a global variable or a class attribute for simplicity in this example
    # or unittest.mock.Mock for more robust testing.
    
    call_count_dummy_func = 0

    def dummy_data_func_for_test(self, arg1, arg2="default"):
        TestCacheFunctionality.call_count_dummy_func += 1
        return {"arg1": arg1, "arg2": arg2, "count": TestCacheFunctionality.call_count_dummy_func}

    def reset_dummy_func_call_count(self):
        TestCacheFunctionality.call_count_dummy_func = 0
        
    def test_cached_func_default_dir(self):
        self.reset_dummy_func_call_count()
        
        # Decorate the function. Key is optional, but good for differentiating tests.
        # No 'args' here anymore, they are passed at call time.
        @cache.cached_func(key="default_dir_test")
        def decorated_dummy_func(arg1, arg2="default_val"):
            return self.dummy_data_func_for_test(arg1, arg2=arg2)

        # Call multiple times with arguments
        result1 = decorated_dummy_func("param1_val", arg2="param2_val")
        result2 = decorated_dummy_func("param1_val", arg2="param2_val") # Should be cached

        self.assertEqual(TestCacheFunctionality.call_count_dummy_func, 1) 
        self.assertEqual(result1, result2) 
        self.assertEqual(result1["arg1"], "param1_val")
        self.assertEqual(result1["arg2"], "param2_val")
        
        # Verify cache file in default directory (_config.TMP_DIR is mocked)
        # The key for cached_func now includes function name, decorator key, and call-time args
        call_specific_key_parts = ["default_dir_test", "param1_val", "arg2=param2_val"]
        combined_key_str = "_".join(call_specific_key_parts)
        md5_key_part = cache.quick_md5_of_object(combined_key_str)
        cache_file_name = f"decorated_dummy_func_{md5_key_part}.cache" # func name is 'decorated_dummy_func'
        expected_cache_file = os.path.join(self.default_test_cache_dir, cache_file_name)
        self.assertTrue(os.path.exists(expected_cache_file))

    def test_cached_func_custom_dir(self):
        self.reset_dummy_func_call_count()
        custom_cache_path_for_decorator = os.path.join(self.temp_dir_root, "decorator_custom_cache")
        
        @cache.cached_func(key="custom_dir_test", cache_dir=custom_cache_path_for_decorator)
        def decorated_dummy_func_custom(arg1, kwarg_test=None):
            return self.dummy_data_func_for_test(arg1, arg2=kwarg_test if kwarg_test is not None else "default_kwarg")

        # Call multiple times
        result1 = decorated_dummy_func_custom("custom_param1", kwarg_test="custom_kwarg")
        result2 = decorated_dummy_func_custom("custom_param1", kwarg_test="custom_kwarg") # Should be cached

        self.assertEqual(TestCacheFunctionality.call_count_dummy_func, 1)
        self.assertEqual(result1, result2)
        self.assertEqual(result1["arg1"], "custom_param1")
        self.assertEqual(result1["arg2"], "custom_kwarg")

        call_specific_key_parts = ["custom_dir_test", "custom_param1", "kwarg_test=custom_kwarg"]
        combined_key_str = "_".join(call_specific_key_parts)
        md5_key_part = cache.quick_md5_of_object(combined_key_str)
        cache_file_name = f"decorated_dummy_func_custom_{md5_key_part}.cache"
        expected_cache_file = os.path.join(custom_cache_path_for_decorator, cache_file_name)
        self.assertTrue(os.path.exists(expected_cache_file))
        
        # Ensure it's not in the default dir
        default_dir_file_path = os.path.join(self.default_test_cache_dir, cache_file_name)
        self.assertFalse(os.path.exists(default_dir_file_path))

    @patch('quick_vllm.api.arg_dict', MagicMock(get=lambda key, default: {"use_cache": 1, "disable_cache": 0}.get(key, default)))
    @patch('quick_vllm.api.get_mdl') # Mock get_mdl
    @patch('quick_vllm.api.get_client') # Mocks get_client to avoid actual API calls
    def test_api_send_custom_cache(self, mock_get_client, mock_get_mdl_api): # Mocks are applied bottom-up
        # Setup mock client and its behavior
        mock_get_mdl_api.return_value = "mocked_model_name" # Mock model name for api.get_mdl()
        mock_client_instance = MagicMock()
        mock_completion_instance = MagicMock()
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.delta = MagicMock(content=" Test response") # Streamed response
        mock_choice.message.content = "Test response" # Non-streamed response
        mock_choice.finish_reason = "stop"
        mock_choice.stop_reason = None # for stream=False

        # For stream=True
        def create_completion_stream(*args, **kwargs):
            # Simulate streaming chunks
            mock_chunk = MagicMock()
            mock_chunk.choices = [mock_choice]
            return iter([mock_chunk, mock_chunk]) # Return a couple of chunks

        # For stream=False
        mock_completion_instance.choices = [mock_choice]


        # Based on how _run_message calls create, decide if stream=True or stream=False is default
        # The example calls send(), which calls _batch_send_message_wrapper, then send_message, then _run_message
        # _run_message has stream=True by default.
        mock_client_instance.chat.completions.create.side_effect = create_completion_stream
        mock_get_client.return_value = mock_client_instance

        custom_api_cache_dir = os.path.join(self.temp_dir_root, "api_custom_cache")
        messages_to_send = ["Hello VLLM!"]
        
        # First call - should call API and create cache
        api.send(messages_to_send, cache_dir=custom_api_cache_dir, temperature=0.1) # Added some sampling param for stable hashing

        # Check if cache file was created in custom_api_cache_dir
        # The filename is a hash of settings, so we just check if *any* .cache file exists
        cache_files = [f for f in os.listdir(custom_api_cache_dir) if f.endswith(".cache")]
        self.assertTrue(len(cache_files) > 0, "Cache file was not created in custom API cache directory.")
        
        # Reset mock call count for the actual API call part
        mock_client_instance.chat.completions.create.reset_mock()

        # Second call with same parameters - should use cache, not call API
        api.send(messages_to_send, cache_dir=custom_api_cache_dir, temperature=0.1)
        
        # Assert that the mocked API (client.chat.completions.create) was NOT called this time
        mock_client_instance.chat.completions.create.assert_not_called()

    def test_async_send_returns_same_result(self):
        messages = ["a", "b", "c"]

        def fake_wrapper(d):
            return f"resp_{d['msg']}"

        with patch("quick_vllm.api._batch_send_message_wrapper", side_effect=fake_wrapper), \
             patch("multiprocessing.Pool", mp.pool.ThreadPool):
            sync_result = api.send(messages)
            handle = api.send(messages, async_=True)
            async_result = handle.get()
            async_result_2 = api.send_async(messages).get()

        self.assertEqual(async_result, sync_result)
        self.assertEqual(async_result_2, sync_result)
        self.assertEqual(async_result, [f"resp_{m}" for m in messages])

    def test_async_send_individual_get(self):
        messages = ["a", "b", "c"]

        def fake_wrapper(d):
            return f"resp_{d['msg']}"

        with patch("quick_vllm.api._batch_send_message_wrapper", side_effect=fake_wrapper), \
             patch("multiprocessing.Pool", mp.pool.ThreadPool):
            handle = api.send_async(messages)
            results = [item.get() for item in handle]

        self.assertEqual(results, [f"resp_{m}" for m in messages])

    def test_client_async_stream_print(self):
        messages = ["hi"]

        def fake_worker(d):
            assert d["kwargs"]["stream"] is True
            print("TOK", flush=True)
            return f"resp_{d['msg']}"

        with patch("quick_vllm.vllm_client._worker_send_wrapper", side_effect=fake_worker), \
             patch("multiprocessing.Pool", side_effect=AssertionError("Proc used")), \
             patch("quick_vllm.vllm_client.VLLMClient._model_id", return_value="mdl"):
            client = vllm_client.VLLMClient()
            with io.StringIO() as buf, redirect_stdout(buf):
                handle = client.send_async(messages, stream_print=True)
                result = handle.get()
                output = buf.getvalue()

        self.assertIn("TOK", output)
        self.assertEqual(result, ["resp_hi"])


if __name__ == '__main__':
    unittest.main()
