.PHONY: all

all: raw

raw: raw/train.csv raw/test.csv

raw/train.csv:
	mkdir -p raw
	gdown https://drive.google.com/uc?id=15MZ9jpjBUlBZQ4RMkOtWKfZc3MtwSDHO -O raw/train.csv

raw/test.csv:
	mkdir -p raw
	gdown https://drive.google.com/uc?id=1HouuT3oVsJtlGZnOhj70TFFBad0EZIRe -O raw/test.csv