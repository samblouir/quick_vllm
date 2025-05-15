
import json
import quick_vllm.api as qvapi
from quick_vllm.metrics import compute_bleu

# Example usage
msgs = [
	"Translate English to Spanish. Do not generate any other text. ```Wow! We can hear the waterfall.```",
]

labels = [
	[
		"¡Hola! ¡Podemos escuchar la cascada!",
		"¡Guau! ¡Podemos escuchar la cascada!",
		"¡Guau! ¡Podemos oír la cascada!",
		"¡Guau! ¡Podemos escuchar la catarata!",
		"¡Guau! ¡Podemos oír la catarata!",
	], # List of possible correct translations
]

just_return_text = 1

message_responses = qvapi.send(
	msgs,
	just_return_text=just_return_text,
	temperature=0.7,
	top_p=0.95,
	min_p=0.0,
	n=64, # Generate 4 responses per message
)

for idx in range(len(message_responses)):
	current_response = message_responses[idx]
	current_labels = labels[idx]

	for current_response_idx, (_current_response) in enumerate(current_response):
		print(f"  current_response[{current_response_idx}]: {_current_response}")
		

	bleu_scores = compute_bleu(
		candidates=current_response,
		references=current_labels,
	)

	print(f"\n" * 3, end='',)
	print(f"#" * 48,)
	max_bleu_score = 0.0
	best_entry = bleu_scores[0]

	for i, bleu_entry in enumerate(bleu_scores):
		if bleu_entry["bleu_score"] > max_bleu_score:
			max_bleu_score = bleu_entry["bleu_score"]
			best_entry = bleu_entry

		print(f"*" * 60,)
		print(f"BLEU Score {i}:")
		print(f"  bleu_score: {bleu_entry['bleu_score']}")
		print(f"  reference: {bleu_entry['reference']}")
		print(f"  candidate: {bleu_entry['candidate']}")

	print(f"*" * 60,)
	print(f"Best BLEU Score: {max_bleu_score:0.2f}")
	print(f"  best reference: {best_entry['reference']}")
	print(f"  best candidate: {best_entry['candidate']}")

	