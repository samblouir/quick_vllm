import os

def get_TMP_DIR():
	"""
	Creates a temporary directory for storing files.
	"""
		
	filepath_to_this_file = os.path.abspath(__file__)
	TMP_DIR = os.path.join(os.path.dirname(filepath_to_this_file), "tmp")
	os.makedirs(TMP_DIR, exist_ok=True)
	return TMP_DIR

TMP_DIR = get_TMP_DIR()
