import sys

arg_dict = {
	"disable_cache": 1,
}
for arg in sys.argv:
	if "=" in arg:
		arg_name, arg_value = arg.split("=")
		try:
			arg_value = int(arg_value)
		except:
			try:
				arg_value = float(arg_value)
			except:
				pass
		arg_dict[arg_name] = arg_value
pass

def cleaner(x):
	x = x.strip()
	trd = {
		"**": "",
		"    ": "\t",
		"  ": " ",
	}
	for k, v in trd.items():
		while k in x:
			x = x.replace(k, v)
	return x.strip()
pass

def get_boxxed(x):
	outputs = {}
	for key, value in x.items():
		if isinstance(value, list):
			value = ''.join(value)
		if "<box>" in value and "</box>" in value:
			value = value.rsplit("<box>", 1)[-1].rsplit("</box>", 1)[0].strip()
		else:
			value = ""
		outputs[key] = value
	return outputs
pass


def extract_boxxed(value):
	# print(f"  value: {value}")
	if isinstance(value, dict):
		try:
			value = value['text']
		except Exception as e:
			print(f"  EXCEPTION IN {__file__}.extract_boxxed():  value['text'].  value.keys(): {value.keys()}, error: {e}")
			# keys = value.keys()
			# for keys_idx, (_keys) in enumerate(keys):
			# 	print(f"  value.keys()[{keys_idx}]: {_keys}")
				
			raise e

	if "<box>" in value and "</box>" in value:
		value = value.rsplit("<box>", 1)[-1].rsplit("</box>", 1)[0].strip()
	else:
		value = ""
	
	value = cleaner(value)
	return value
pass


def extract_thinking(value):
	if "<think>" in value and "</think>" in value:
		value = value.rsplit("<think>", 1)[-1].rsplit("</think>", 1)[0].strip()
	else:
		value = ""
	return value
pass


def extract_other(value):
	if "</think>" in value:
		value = value.rsplit("</think>", 1)[-1].strip()
	
	prebox = value.rsplit("<box>", 1)[0].strip()	
	postbox = value.rsplit("</box>", 1)[-1].strip()

	other_text = prebox + "<box_placeholder>" + postbox
	return other_text
pass
