.PHONY: data install help model

help:
	@echo "download - download data"
	@echo "install - install dependencies"
	@echo "model - train model"

install:
	poetry install --only main --no-root

download:
	$(MAKE) -C data

model:
	poetry run python src/main.py model
