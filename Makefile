PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: test doctor

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) -c "from renquant_common import Pipeline, Task, Job; print('renquant-common ok')"
