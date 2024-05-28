.PHONY: all

all: download

download: raw/train.csv raw/test.csv

raw/train.csv:
ifeq ($(OS),Windows_NT)
	if not exist "raw" mkdir raw
	poetry run gdown https://drive.google.com/uc?id=15MZ9jpjBUlBZQ4RMkOtWKfZc3MtwSDHO -o raw/train.csv
else
	mkdir -p raw
	poetry run gdown https://drive.google.com/uc?id=15MZ9jpjBUlBZQ4RMkOtWKfZc3MtwSDHO -O raw/train.csv
endif

raw/test.csv:
ifeq ($(OS),Windows_NT)
	if not exist "raw" mkdir raw
	poetry run gdown https://drive.google.com/uc?id=1HouuT3oVsJtlGZnOhj70TFFBad0EZIRe -o raw/test.csv
else
	mkdir -p raw
	poetry run gdown https://drive.google.com/uc?id=1HouuT3oVsJtlGZnOhj70TFFBad0EZIRe -O raw/test.csv
endif