.PHONY: setup test lint report run model

setup:
	python -m pip install -r requirements.txt

test:
	python -m pytest

lint:
	python -m ruff check .

report:
	python -m src.report

model:
	python scripts/download_model.py

run:
	python -m streamlit run app.py

