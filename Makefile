.PHONY: data

install:
	poetry install --no-root

data:
	$(MAKE) -C data
