import re

import numpy as np

NUM_PAT = re.compile("(\\-?[0-9\\.\\,]+)")


def extract_numeric(
	text, 
	choices=[], 
	aliases=[], 
	take_extraction_pos=0, 
	as_string=False,
	**kwargs
):
	
	if numeric := NUM_PAT.findall(text):
		numeric = numeric[take_extraction_pos]
	else:
		numeric = ""

	# strip away extra / dangling decimals e.g. due text end punctuation
	decimal_splits = re.split(r"\.", numeric)
	if len(decimal_splits) >= 2:
		numeric = ".".join(decimal_splits[:2])

	if not as_string:
		try:
			numeric = float(numeric.replace(",", ""))
		except ValueError as e:
			numeric = np.nan

	return numeric
