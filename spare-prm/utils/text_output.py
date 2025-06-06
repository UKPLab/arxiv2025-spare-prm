import re

# list of abbreviations with periods from cmudict
# https://github.com/cmusphinx/cmudict
PERIOD_ABBR = [
	"a.d.",
	"a.m.",
	"al.",
	"b.c.",
	"c.o.d.",
	"co.",
	"conn.",
	"corp.",
	"cr.",
	"d.c.",
	"dr.",
	"e.g.",
	"etc.",
	"in.",
	"inc.",
	"jan.",
	"jr.",
	"ltd.",
	"m.d.",
	"mass.",
	"messrs.",
	"mr.",
	"mrs.",
	"ms.",
	"mssrs.",
	"p.m.",
	"ph.d.",
	"prof.",
	"rep.",
	"u.k.",
	"u.n.",
	"u.s.",
	"u.s.a.",
	"u.s.c.",
	"v.a.",
	"vs.",
]


def extract_clean_text(text, choices=[], aliases=[], **kwargs):
	def remove_quotes_encapsulation(t):
		if t and t[-1] == "\"":
			t = t[:-1]
		if t and t[0] == "\"":
			t = t[1:]
		return t

	text = remove_quotes_encapsulation(text.strip())
	
	last_token = re.split(r"\s+", text)[-1].lower()
	if last_token and last_token[-1] == "." and last_token not in PERIOD_ABBR:
		text = text[:-1]

	text = remove_quotes_encapsulation(text)

	return text
