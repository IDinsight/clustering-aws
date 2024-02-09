# Load project config
include ./project_config.cfg
export

SHELL = /bin/bash

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

fresh-env:
	conda remove --name $(PROJECT_CONDA_ENV) --all -y
	conda create --name $(PROJECT_CONDA_ENV) python==3.11 -y
	
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); \
	pip install --upgrade pip \
	pip install -e . \
	pip install -r requirements_dev.txt --ignore-installed \
	pre-commit install

clean:
	find .. -type f -name "*.py[co]" -delete
	find .. -type d -name "__pycache__" -delete
	find .. -type d -name ".pytest_cache" -delete