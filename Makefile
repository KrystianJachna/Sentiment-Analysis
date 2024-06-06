.PHONY: data install help model gui

help:
	@echo "download - download data & model"
	@echo "install - install dependencies"
	@echo "model - train model"

install:
	poetry install --only main --no-root
ifeq ($(OS),Windows_NT)
	if not exist "cache" mkdir cache
	if not exist "flagged" mkdir flagged
else
	mkdir -p cache flagged
endif

download:
	poetry run gdown https://drive.google.com/uc?id=1EtBwEizusjGr3dzvRqJ0uxmESyGxrhDr -O model/model.pkl
	$(MAKE) -C data

model:
	poetry run python src/main.py model

gui:
	poetry run python src/main.py gui
