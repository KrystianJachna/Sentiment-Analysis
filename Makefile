.PHONY: data install help model gui prediction

help:
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  help       - Show this help message"
	@echo "  download   - Download data & model from Google Drive"
	@echo "  install    - Install dependencies from pyproject.toml and download nltk data"
	@echo "  gui        - Run gradio-GUI using the model: model/model.pkl"
	@echo "  model      - Train and test the model using: data/train.csv, data/test.csv. Save the model to model/model.pkl"
	@echo "  prediction - Predict the sentiment of a given review using model/model.pkl."
	@echo "               Example: make prediction review='This is a great product!'"

install:
	poetry install --only main --no-root
ifeq ($(OS),Windows_NT)
	if not exist "cache" mkdir cache
	if not exist "flagged" mkdir flagged
else
	mkdir -p cache flagged
endif
	poetry run python -c "import nltk; nltk.download('punkt')"

download:
	poetry run gdown https://drive.google.com/uc?id=1EtBwEizusjGr3dzvRqJ0uxmESyGxrhDr -O model/model.pkl
	$(MAKE) -C data

model:
	poetry run python src/main.py model

gui:
	poetry run python src/main.py gui

prediction:
	poetry run python src/main.py predict --review="$(review)"
