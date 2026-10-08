#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = teplayakov_an_mlops_project
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
APP_HOST ?= 127.0.0.1
APP_PORT ?= 8000
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
MLFLOW_BACKEND_STORE_URI ?= sqlite:///artifacts/mlflow/mlflow.db
MLFLOW_ARTIFACTS_DESTINATION ?= $(CURDIR)/artifacts/mlflow/mlartifacts
MLFLOW_TRACKING_URI ?= http://127.0.0.1:$(MLFLOW_PORT)
MODELS ?= log_reg random_forest catboost
N_JOBS ?= -1
DATA_PATH ?= data/processed/UCI_Credit_Card.csv

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 teplayakov_an_mlops_project
	isort --check --diff teplayakov_an_mlops_project
	black --check teplayakov_an_mlops_project

## Format source code with black
.PHONY: format
format:
	isort teplayakov_an_mlops_project
	black teplayakov_an_mlops_project



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Download dataset into data/raw
.PHONY: data
data:
	uv run python -m src.data.make_dataset

## validate and clean the dataset into data/processed
.PHONY: prepare-data 
prepare-data:
	uv run python -m src.data.prepare_dataset

## Train and register the best credit default model
.PHONY: train
train:
	uv run python -m src.model_training.train

## Start MLflow UI and tracking API (localhost:5000 by default)
.PHONY: mlflow
mlflow:
	mkdir -p artifacts/mlflow
	uv run mlflow server --host "$(MLFLOW_HOST)" --port "$(MLFLOW_PORT)" \
		--backend-store-uri "$(MLFLOW_BACKEND_STORE_URI)" \
		--artifacts-destination "$(MLFLOW_ARTIFACTS_DESTINATION)"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
