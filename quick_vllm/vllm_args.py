
'''

Dump of various VLLM arguments and settings.


(method) def create(
	*,
	messages: Iterable[ChatCompletionMessageParam],
	model: ChatModel | str,
	audio: ChatCompletionAudioParam | NotGiven | None = NOT_GIVEN,
	frequency_penalty: float | NotGiven | None = NOT_GIVEN,
	function_call: FunctionCall | NotGiven = NOT_GIVEN,
	functions: Iterable[Function] | NotGiven = NOT_GIVEN,
	logit_bias: Dict[str, int] | NotGiven | None = NOT_GIVEN,
	logprobs: bool | NotGiven | None = NOT_GIVEN,
	max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
	max_tokens: int | NotGiven | None = NOT_GIVEN,
	metadata: Dict[str, str] | NotGiven | None = NOT_GIVEN,
	modalities: List[ChatCompletionModality] | NotGiven | None = NOT_GIVEN,
	n: int | NotGiven | None = NOT_GIVEN,
	parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
	prediction: ChatCompletionPredictionContentParam | NotGiven | None = NOT_GIVEN,
	presence_penalty: float | NotGiven | None = NOT_GIVEN,
	reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
	response_format: ResponseFormat | NotGiven = NOT_GIVEN,
	seed: int | NotGiven | None = NOT_GIVEN,
	service_tier: NotGiven | Literal['auto', 'default'] | None = NOT_GIVEN,
	stop: str | List[str] | NotGiven | None = NOT_GIVEN,
	store: bool | NotGiven | None = NOT_GIVEN,
	stream: NotGiven | Literal[False] | None = NOT_GIVEN,
	stream_options: ChatCompletionStreamOptionsParam | NotGiven | None = NOT_GIVEN,
	temperature: float | NotGiven | None = NOT_GIVEN,
	tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
	tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
	top_logprobs: int | NotGiven | None = NOT_GIVEN,
	top_p: float | NotGiven | None = NOT_GIVEN,
	user: str | NotGiven = NOT_GIVEN,
	extra_headers: Headers | None = None,
	extra_query: Query | None = None,
	extra_body: Body | None = None,
	timeout: float | Timeout | NotGiven | None = NOT_GIVEN
) -> ChatCompletion

docker run --runtime nvidia --gpus all \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
	--env "HUGGING_FACE_HUB_TOKEN=<secret>" \
	-p 8000:8000 \
	--ipc=host \
	vllm/vllm-openai:latest \
	--model deephermes-3-llama-3-8b-preview

	usage: api_server.py [-h] [--host HOST] [--port PORT]
					 [--uvicorn-log-level {debug,info,warning,error,critical,trace}]
					 [--allow-credentials] [--allowed-origins ALLOWED_ORIGINS]
					 [--allowed-methods ALLOWED_METHODS]
					 [--allowed-headers ALLOWED_HEADERS] [--api-key API_KEY]
					 [--lora-modules LORA_MODULES [LORA_MODULES ...]]
					 [--prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]]
					 [--chat-template CHAT_TEMPLATE]
					 [--response-role RESPONSE_ROLE]
					 [--ssl-keyfile SSL_KEYFILE] [--ssl-certfile SSL_CERTFILE]
					 [--ssl-ca-certs SSL_CA_CERTS]
					 [--ssl-cert-reqs SSL_CERT_REQS] [--root-path ROOT_PATH]
					 [--middleware MIDDLEWARE] [--return-tokens-as-token-ids]
					 [--disable-frontend-multiprocessing] [--model MODEL]
					 [--tokenizer TOKENIZER] [--skip-tokenizer-init]
					 [--revision REVISION] [--code-revision CODE_REVISION]
					 [--tokenizer-revision TOKENIZER_REVISION]
					 [--tokenizer-mode {auto,slow}] [--trust-remote-code]
					 [--download-dir DOWNLOAD_DIR]
					 [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,bitsandbytes}]
					 [--dtype {auto,half,float16,bfloat16,float,float32}]
					 [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
					 [--quantization-param-path QUANTIZATION_PARAM_PATH]
					 [--max-model-len MAX_MODEL_LEN]
					 [--guided-decoding-backend {outlines,lm-format-enforcer}]
					 [--distributed-executor-backend {ray,mp}]
					 [--worker-use-ray]
					 [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
					 [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
					 [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
					 [--ray-workers-use-nsight] [--block-size {8,16,32}]
					 [--enable-prefix-caching] [--disable-sliding-window]
					 [--use-v2-block-manager]
					 [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS] [--seed SEED]
					 [--swap-space SWAP_SPACE]
					 [--cpu-offload-gb CPU_OFFLOAD_GB]
					 [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
					 [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE]
					 [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
					 [--max-num-seqs MAX_NUM_SEQS]
					 [--max-logprobs MAX_LOGPROBS] [--disable-log-stats]
					 [--quantization {aqlm,awq,deepspeedfp,fp8,fbgemm_fp8,marlin,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,squeezellm,compressed-tensors,bitsandbytes,qqq,None}]
					 [--rope-scaling ROPE_SCALING] [--rope-theta ROPE_THETA]
					 [--enforce-eager]
					 [--max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE]
					 [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE]
					 [--disable-custom-all-reduce]
					 [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
					 [--tokenizer-pool-type TOKENIZER_POOL_TYPE]
					 [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG]
					 [--enable-lora] [--max-loras MAX_LORAS]
					 [--max-lora-rank MAX_LORA_RANK]
					 [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE]
					 [--lora-dtype {auto,float16,bfloat16,float32}]
					 [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS]
					 [--max-cpu-loras MAX_CPU_LORAS] [--fully-sharded-loras]
					 [--enable-prompt-adapter]
					 [--max-prompt-adapters MAX_PROMPT_ADAPTERS]
					 [--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN]
					 [--device {auto,cuda,neuron,cpu,openvino,tpu,xpu}]
					 [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR]
					 [--enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]]
					 [--speculative-model SPECULATIVE_MODEL]
					 [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
					 [--speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE]
					 [--speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN]
					 [--speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE]
					 [--ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX]
					 [--ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN]
					 [--spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}]
					 [--typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD]
					 [--typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA]
					 [--disable-logprobs-during-spec-decoding DISABLE_LOGPROBS_DURING_SPEC_DECODING]
					 [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
					 [--ignore-patterns IGNORE_PATTERNS]
					 [--preemption-mode PREEMPTION_MODE]
					 [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]]
					 [--qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH]
					 [--otlp-traces-endpoint OTLP_TRACES_ENDPOINT]
					 [--engine-use-ray] [--disable-log-requests]
					 [--max-log-len MAX_LOG_LEN]


					 [--allow-credentials] [--allowed-origins ALLOWED_ORIGINS]
					 [--allowed-headers ALLOWED_HEADERS] [--api-key API_KEY]
					 [--allowed-methods ALLOWED_METHODS]
					 [--chat-template CHAT_TEMPLATE]
					 [--cpu-offload-gb CPU_OFFLOAD_GB]
					 [--device {auto,cuda,neuron,cpu,openvino,tpu,xpu}]
					 [--disable-custom-all-reduce]
					 [--disable-frontend-multiprocessing] [--model MODEL]
					 [--disable-logprobs-during-spec-decoding DISABLE_LOGPROBS_DURING_SPEC_DECODING]
					 [--distributed-executor-backend {ray,mp}]
					 [--download-dir DOWNLOAD_DIR]
					 [--dtype {auto,half,float16,bfloat16,float,float32}]
					 [--enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]]
					 [--enable-lora] [--max-loras MAX_LORAS]
					 [--enable-prefix-caching] [--disable-sliding-window]
					 [--enable-prompt-adapter]
					 [--enforce-eager]
					 [--engine-use-ray] [--disable-log-requests]
					 [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
					 [--guided-decoding-backend {outlines,lm-format-enforcer}]
					 [--ignore-patterns IGNORE_PATTERNS]
					 [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
					 [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,bitsandbytes}]
					 [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS]
					 [--lora-dtype {auto,float16,bfloat16,float32}]
					 [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE]
					 [--lora-modules LORA_MODULES [LORA_MODULES ...]]
					 [--max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE]
					 [--max-cpu-loras MAX_CPU_LORAS] [--fully-sharded-loras]
					 [--max-log-len MAX_LOG_LEN]
					 [--max-logprobs MAX_LOGPROBS] [--disable-log-stats]
					 [--max-lora-rank MAX_LORA_RANK]
					 [--max-model-len MAX_MODEL_LEN]
					 [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
					 [--max-num-seqs MAX_NUM_SEQS]
					 [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
					 [--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN]
					 [--max-prompt-adapters MAX_PROMPT_ADAPTERS]
					 [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE]
					 [--middleware MIDDLEWARE] [--return-tokens-as-token-ids]
					 [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
					 [--ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX]
					 [--ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN]
					 [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE]
					 [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS] [--seed SEED]
					 [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
					 [--otlp-traces-endpoint OTLP_TRACES_ENDPOINT]
					 [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
					 [--preemption-mode PREEMPTION_MODE]
					 [--prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]]
					 [--qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH]
					 [--quantization {aqlm,awq,deepspeedfp,fp8,fbgemm_fp8,marlin,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,squeezellm,compressed-tensors,bitsandbytes,qqq,None}]
					 [--quantization-param-path QUANTIZATION_PARAM_PATH]
					 [--ray-workers-use-nsight] [--block-size {8,16,32}]
					 [--response-role RESPONSE_ROLE]
					 [--revision REVISION] [--code-revision CODE_REVISION]
					 [--rope-scaling ROPE_SCALING] [--rope-theta ROPE_THETA]
					 [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR]
					 [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]]
					 [--spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}]
					 [--speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE]
					 [--speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE]
					 [--speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN]
					 [--speculative-model SPECULATIVE_MODEL]
					 [--ssl-ca-certs SSL_CA_CERTS]
					 [--ssl-cert-reqs SSL_CERT_REQS] [--root-path ROOT_PATH]
					 [--ssl-keyfile SSL_KEYFILE] [--ssl-certfile SSL_CERTFILE]
					 [--swap-space SWAP_SPACE]
					 [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
					 [--tokenizer TOKENIZER] [--skip-tokenizer-init]
					 [--tokenizer-mode {auto,slow}] [--trust-remote-code]
					 [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG]
					 [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
					 [--tokenizer-pool-type TOKENIZER_POOL_TYPE]
					 [--tokenizer-revision TOKENIZER_REVISION]
					 [--typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA]
					 [--typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD]
					 [--use-v2-block-manager]
					 [--uvicorn-log-level {debug,info,warning,error,critical,trace}]
					 [--worker-use-ray]


	VLLM SamplingParams(
		n=8
		best_of=8
		presence_penalty=0.0
		frequency_penalty=0.0
		repetition_penalty=1.0
		temperature=0.7
		top_p=1.0
		top_k=-1
		min_p=0.0
		seed=None
		use_beam_search=False
		length_penalty=1.0
		early_stopping=False
		stop=[]
		stop_token_ids=[]
		include_stop_str_in_output=False
		ignore_eos=False
		max_tokens=131035
		min_tokens=0
		logprobs=None
		prompt_logprobs=None
		skip_special_tokens=True
		spaces_between_special_tokens=True
		truncate_prompt_tokens=None
	)

	OpenAI Settings
	messages: Iterable[ChatCompletionMessageParam],
	model: ChatModel | str,
	audio: ChatCompletionAudioParam | NotGiven | None = NOT_GIVEN,
	frequency_penalty: float | NotGiven | None = NOT_GIVEN,
	function_call: FunctionCall | NotGiven = NOT_GIVEN,
	functions: Iterable[Function] | NotGiven = NOT_GIVEN,
	logit_bias: Dict[str, int] | NotGiven | None = NOT_GIVEN,
	logprobs: bool | NotGiven | None = NOT_GIVEN,
	max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
	max_tokens: int | NotGiven | None = NOT_GIVEN,
	metadata: Dict[str, str] | NotGiven | None = NOT_GIVEN,
	modalities: List[ChatCompletionModality] | NotGiven | None = NOT_GIVEN,
	n: int | NotGiven | None = NOT_GIVEN,
	parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
	prediction: ChatCompletionPredictionContentParam | NotGiven | None = NOT_GIVEN,
	presence_penalty: float | NotGiven | None = NOT_GIVEN,
	reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
	response_format: ResponseFormat | NotGiven = NOT_GIVEN,
	seed: int | NotGiven | None = NOT_GIVEN,
	service_tier: NotGiven | Literal['auto', 'default'] | None = NOT_GIVEN,
	stop: str | List[str] | NotGiven | None = NOT_GIVEN,
	store: bool | NotGiven | None = NOT_GIVEN,
	stream: NotGiven | Literal[False] | None = NOT_GIVEN,
	stream_options: ChatCompletionStreamOptionsParam | NotGiven | None = NOT_GIVEN,
	temperature: float | NotGiven | None = NOT_GIVEN,
	tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
	tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
	top_logprobs: int | NotGiven | None = NOT_GIVEN,
	top_p: float | NotGiven | None = NOT_GIVEN,
	user: str | NotGiven = NOT_GIVEN,
	extra_headers: Headers | None = None,
	extra_query: Query | None = None,
	extra_body: Body | None = None,
	timeout: float | Timeout | NotGiven | None = NOT_GIVEN
'''
