PY=python

.PHONY: install test lint run-real run-app docker-build docker-run

install:
	$(PY) -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

lint:
	ruff .

run-real:
	$(PY) run.py real --tickers AAPL,MSFT --rolling 36

run-app:
	streamlit run streamlit_app.py

docker-build:
	docker build -t pfa:latest .

docker-run:
	docker run --rm -p 8501:8501 pfa:latest

