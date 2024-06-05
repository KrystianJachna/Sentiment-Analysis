.PHONY: all

all: download

download: raw/train.csv raw/test.csv

raw/train.csv:
	poetry run gdown https://drive.google.com/uc?id=15MZ9jpjBUlBZQ4RMkOtWKfZc3MtwSDHO -O train.csv

raw/test.csv:
	poetry run gdown https://drive.google.com/uc?id=1HouuT3oVsJtlGZnOhj70TFFBad0EZIRe -O test.csv
