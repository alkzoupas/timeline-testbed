# timeline-testbed: developer tasks. test, lint and typecheck run offline (AGENTS.md).

VENV   := .venv
PY     := $(VENV)/bin/python
SITE   := $(VENV)/lib/python3.14/site-packages
VENDOR := vendor/entitlement-shield

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  venv            create .venv from requirements.txt (an install: needs the owner's go)"
	@echo "  test            pytest over tests/"
	@echo "  lint            ruff over testbed/ and tests/"
	@echo "  typecheck       mypy over testbed/ and tests/"
	@echo "  vendor-check    verify vendor/ against VENDORED.sha256"
	@echo "  vendored-tests  run the vendored suite once on a temporary copy (M0 baseline)"
	@echo "  timing          time generate/persist/detect/units on the vendored min and dev presets"

# The vendored package is put on the path, not installed, so nothing is written into vendor/.
# The .pth also redirects bytecode caches to .venv/pycache.
.PHONY: venv
venv:
	uv venv $(VENV) --python python3.14
	uv pip sync requirements.txt --python $(PY)
	printf '%s\n%s\n%s\n' "$(CURDIR)" "$(CURDIR)/$(VENDOR)/backend" \
		"import sys; sys.pycache_prefix = sys.pycache_prefix or '$(CURDIR)/$(VENV)/pycache'" \
		> $(SITE)/timeline_testbed.pth

.PHONY: test
test:
	$(PY) -m pytest

.PHONY: lint
lint:
	$(PY) -m ruff check testbed tests

.PHONY: typecheck
typecheck:
	$(PY) -m mypy

.PHONY: vendor-check
vendor-check:
	cd $(VENDOR) && shasum -a 256 -c VENDORED.sha256 --quiet

# Runs on a copy in $TMPDIR (left for the OS to clean), with API-key variables unset, so the
# suite can't write into vendor/ or call a hosted API. Tests that need the predecessor's data/
# fail or skip; that's the baseline.
.PHONY: vendored-tests
vendored-tests:
	@tmp=$$(mktemp -d) && echo "copy: $$tmp/es" && cp -R $(VENDOR) $$tmp/es && cd $$tmp/es && \
		env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY $(CURDIR)/$(PY) -m pytest -q -p no:cacheprovider

.PHONY: timing
timing:
	$(PY) -m testbed.pipeline.timing --preset min --out runs/m0-timing
	$(PY) -m testbed.pipeline.timing --preset dev --out runs/m0-timing
