## Top-level developer entry point. See docs/spec/02-public-api.md §Makefile targets
## and RFC-0006 §D7. Every target is non-recursive and idempotent.

PYTHON ?= python
UV ?= uv

.DEFAULT_GOAL := help
.PHONY: help data pretrain finetune eval eval-docker demo smoke lint test clean-artifacts

help:  ## List every Makefile target.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

data:  ## Run the full Phase 1 data pipeline (mirror → chunk → pairs → dedup → splits → audit → manifest).
	$(PYTHON) -m codingjepa data mirror
	$(PYTHON) -m codingjepa data chunk
	$(PYTHON) -m codingjepa data pairs
	$(PYTHON) -m codingjepa data dedup
	$(PYTHON) -m codingjepa data splits
	$(PYTHON) -m codingjepa data audit
	$(PYTHON) -m codingjepa data manifest

pretrain:  ## Run Stage A (unconditional pretrain). RFC-0008.
	$(PYTHON) -m codingjepa train pretrain --config-name pretrain

finetune:  ## Run Stage B (intent-conditioned fine-tune). RFC-0008.
	$(PYTHON) -m codingjepa train finetune --config-name finetune

eval:  ## Run the full eval harness. RFC-0010.
	$(PYTHON) -m codingjepa eval

eval-docker:  ## Build the reproducible eval image from Dockerfile.eval. See docs/spec/09 §Eval image.
	docker build -f Dockerfile.eval -t codingjepa-eval:test .

demo:  ## Launch the FastAPI + HTMX demo on http://localhost:8080. RFC-0006.
	$(PYTHON) -m codingjepa demo

smoke:  ## Run the eval-smoke 10-example fixture. Exercised by the eval-smoke CI workflow.
	$(PYTHON) -m pytest -m eval-smoke

lint:  ## ruff + black --check + mypy --strict on the package.
	$(UV) run ruff check .
	$(UV) run black --check .
	$(UV) run mypy --strict codingjepa/

test:  ## pytest -m "not slow". The unit CI workflow runs this on CPU.
	$(UV) run pytest -m "not slow"

clean-artifacts:  ## Remove .runs/ and .artifacts/ ONLY. Never touches data/, tokenizer/, runs/, indices/, or checkpoints.
	rm -rf .runs/ .artifacts/
