.PHONY: data install help

help:
	@echo "data - download data"
	@echo "install - install dependencies"

install:
	poetry install --no-root

data:
	$(MAKE) -C data
