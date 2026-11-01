.PHONY: help test lint coverage notebook dev deploy install clean

help:
	@echo "Usage: make <target>"
	@echo "  test          Run all tests"
	@echo "  lint          Check the code with ruff"
	@echo "  coverage      Run the tests with coverage and open the HTML report"
	@echo "  notebook      Re-execute the article's notebook in place"
	@echo "  dev           Build, watch, and serve the article locally"
	@echo "  deploy        Run the tests, then build and deploy to GitHub Pages"
	@echo "  install       Install the package with dev dependencies"

test:
	python -m pytest tests/ -v

lint:
	ruff check .

coverage:
	python -m pytest tests/ --cov=program_tracing --cov-report=term --cov-report=html:coverage-report
	xdg-open coverage-report/index.html

# Re-runs every cell (about two minutes) and regenerates content/widgets/trace-data.js.
notebook:
	cd content && jupyter nbconvert --to notebook --execute --inplace testing-randomized-algorithms.ipynb

dev:
	blog dev

deploy: test
	blog deploy

install:
	pip install -e ".[dev]"

clean:
	rm -rf __pycache__ program_tracing/__pycache__ tests/__pycache__
	rm -rf *.egg-info build dist .pytest_cache coverage-report .coverage
