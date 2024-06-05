.PHONY: all

all: download

download: train.csv test.csv

train.csv:
	poetry run gdown https://drive.google.com/uc?id=15MZ9jpjBUlBZQ4RMkOtWKfZc3MtwSDHO -O train.csv

test.csv:
	poetry run gdown https://drive.google.com/uc?id=1HouuT3oVsJtlGZnOhj70TFFBad0EZIRe -O test.csv
