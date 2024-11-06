#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = skyro
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 skyro
	isort --check --diff --profile black skyro
	black --check --config pyproject.toml skyro

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml skyro

test:
	coverage run -m pytest ./tests

coverage: test
	coverage report --fail-under=90