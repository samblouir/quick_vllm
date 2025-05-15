
import json
import quick_vllm.api as qvapi

# Example usage
msgs = [
	"Hi there, how are you?",
	"The weather is so nice today.",
	# "Hurray for SLP Sidekick!",
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
	n=16, # Generate 16 responses per message
)

if just_return_text:
	for message_responses_idx, (_message_responses) in enumerate(message_responses):
		print(f"  message_responses[{message_responses_idx}]: {_message_responses}")

else:

	##################################################################
	'''
	Print out the raw responses
	'''
	for i, msg_response in enumerate(message_responses):
		print(f"\n" * 3, end='',)
		print(f"#" * 48,)
		print(f"Message Response {i}:")

		# For each `n`, print the result
		for response_idx, (_response) in enumerate(msg_response):

			formatted = json.dumps(_response, indent=2)
			print(f"*" * 60,)
			print(f"  message_responses[{response_idx}]: {formatted}")
			
	print(f"*" * 60,)



	##################################################################
	'''
	Just print out the text from the responses
	'''
	for i, msg_response in enumerate(message_responses):
		print(f"\n" * 3, end='',)
		print(f"#" * 48,)
		print(f"Message Response {i}:")

		input_messages = msg_response[0]["settings"]
		print(f"  input_messages: {json.dumps(input_messages, indent=2)}")


		# For each `n`, print the result
		for response_idx, (_response) in enumerate(msg_response):

			text = _response["text"]
			print(f"*" * 24,)
			print(f"  message_responses[{response_idx}]:\n```\n{text}\n```")
			
	print(f"*" * 60,)
