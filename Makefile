.PHONY: data install help

help:
	@echo "download - download data"
	@echo "install - install dependencies"

install:
	poetry install --only main --no-root

download:
	$(MAKE) -C data
