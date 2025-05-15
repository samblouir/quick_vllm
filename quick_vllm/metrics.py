try:
	from nltk.translate.bleu_score import sentence_bleu
	from nltk.tokenize import word_tokenize
	import nltk
except Exception as e:
	print(f"  WARNING:  metrics.py  nltk not installed!  e: {e}. bleu and other functions will be raise an Exception if called.", flush=True,)

def compute_bleu(reference: str, candidate: str) -> float:
	"""
	Compute the BLEU score between a reference string and a candidate string.
	
	Parameters:
		candidate (str): The candidate text.
		reference (str): The reference text.
		
	Returns:
		float: The BLEU score.
	"""

	reference = reference.lower()
	candidate = candidate.lower()
	trd = {
		"  ": " ",
		" .": ".",
		" ,": ",",
		" !": "!",
		" ?": "?",
		" :": ":",
		" ;": ";",
		" '": "'",
		" \"": "\"",
	}

	for k, v in trd.items():
		while k in reference:
			reference = reference.replace(k, v)
		while k in candidate:
			candidate = candidate.replace(k, v)

	print(f"  reference: {reference}")
	print(f"  candidate: {candidate}")


	# Tokenize both the reference and candidate strings
	try:
		reference_tokens = word_tokenize(reference)
		candidate_tokens = word_tokenize(candidate)
	except:
		nltk.download('punkt')
		nltk.download('punkt_tab')
		reference_tokens = word_tokenize(reference)
		candidate_tokens = word_tokenize(candidate)
	
	# sentence_bleu expects a list of reference sentences (each itself a list of tokens)
	references = [reference_tokens]
	
	# Compute BLEU score
	bleu_score = sentence_bleu(references, candidate_tokens)
	
	return bleu_score

# Example usage:
if __name__ == "__main__":
	ref = "The quick brown fox jumps over the lazy dog."
	cand = "A quick brown fox jumps over the dog."
	score = compute_bleu(ref, cand)
	print(f"BLEU score: {score:.4f}")
