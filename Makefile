.PHONY: install dev test smoke benchmark ablate clean

install:        ## install the core package
	pip install -e .

dev:            ## install with dev + neural extras
	pip install -e ".[dev]"

test:           ## run the test suite
	pytest -q

smoke:          ## fast CPU end-to-end benchmark (no GPU/network/JDK)
	featfuse run -c configs/smoke.yaml

benchmark:      ## reproducible engineered-feature baseline on IR-Plag
	featfuse run -c configs/classical_features_irplag.yaml --ablate

ablate:         ## feature ablation on the baseline config
	featfuse ablate -c configs/classical_features_irplag.yaml

clean:
	rm -rf runs build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
