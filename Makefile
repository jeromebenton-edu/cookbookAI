.RECIPEPREFIX := >
.ONESHELL:
SHELL := /bin/bash

PY=python
VENV?=.venv

BACKEND_PY?=python
BACKEND_HOST?=localhost
BACKEND_BIND_HOST?=0.0.0.0
BACKEND_PORT?=8000
BACKEND_APP?=app.main:app
BACKEND_REQUIREMENTS_FILE:=$(firstword $(wildcard backend/requirements-dev.txt backend/requirements.txt))
FRONTEND_HOST?=localhost
FRONTEND_BIND_HOST?=0.0.0.0
FRONTEND_PORT?=3001
PID_DIR?=.pids

ifeq ($(strip $(BACKEND_REQUIREMENTS_FILE)),)
BACKEND_REQUIREMENTS_FILE:=backend/requirements.txt
endif

ifeq ($(wildcard backend/poetry.lock),backend/poetry.lock)
BACKEND_INSTALL_CMD=cd backend && poetry install
BACKEND_BUILD_CMD=cd backend && poetry run python -m compileall app
BACKEND_RUN_CMD=cd backend && poetry run uvicorn app.main:app --reload --host $(BACKEND_BIND_HOST) --port $(BACKEND_PORT)
else
BACKEND_INSTALL_CMD=PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1 $(BACKEND_PY) -m pip install -r $(BACKEND_REQUIREMENTS_FILE)
BACKEND_BUILD_CMD=$(BACKEND_PY) -m compileall backend/app
BACKEND_RUN_CMD=cd backend && PYTHONPATH=. uvicorn $(BACKEND_APP) --reload --host $(BACKEND_BIND_HOST) --port $(BACKEND_PORT)
endif

ifeq ($(wildcard frontend/pnpm-lock.yaml),frontend/pnpm-lock.yaml)
FRONTEND_PM=pnpm
FRONTEND_INSTALL_CMD=cd frontend && if [ "$${FRONTEND_CLEAN_INSTALL:-0}" = "1" ]; then rm -rf node_modules package-lock.json pnpm-lock.yaml; pnpm install --frozen-lockfile; else pnpm install; fi
FRONTEND_BUILD_CMD=cd frontend && rm -rf .next && pnpm build
FRONTEND_DEV_CMD=cd frontend && pnpm dev -- --hostname $(FRONTEND_BIND_HOST) --port $(FRONTEND_PORT)
else
FRONTEND_PM=npm
FRONTEND_INSTALL_CMD=cd frontend && if [ "$${FRONTEND_CLEAN_INSTALL:-0}" = "1" ]; then rm -rf node_modules package-lock.json; if [ -f package-lock.json ]; then npm ci; else npm install; fi; else npm install; fi
FRONTEND_BUILD_CMD=cd frontend && rm -rf .next && npm run build
FRONTEND_DEV_CMD=cd frontend && npm run dev -- --hostname $(FRONTEND_BIND_HOST) --port $(FRONTEND_PORT)
endif

.PHONY: run-all dev install-backend install-frontend build-backend build-frontend start-backend start-frontend stop health status
.PHONY: backend backend-health backend-stop demo demo-stop smoke hello check-makefile fix-line-endings option2 detect weaklabel build-datasets sanity phase4-run regen-highconf
.PHONY: render-pages build-dataset rebuild-data generate-curated-labels ensure-validation-split build-recipe-only rebuild-recipe-only-250p
.PHONY: clean-frontend frontend-diagnose audit-dataset sanity-labels install-training

$(PID_DIR):
> mkdir -p $(PID_DIR)

install-backend:
> set -euo pipefail
> echo "Installing backend dependencies with $(if $(wildcard backend/poetry.lock),poetry,$(BACKEND_PY))..."
> $(BACKEND_INSTALL_CMD)

install-training: install-backend
> set -euo pipefail
> echo "Verifying training dependencies..."
> $(BACKEND_PY) -c "import accelerate; import seqeval; print('✅ All training dependencies installed')" || \
>   (echo "❌ Missing training dependencies. Installing..."; \
>    $(BACKEND_PY) -m pip install accelerate seqeval; \
>    echo "✅ Training dependencies installed")

install-frontend:
> set -euo pipefail
> echo "Installing frontend dependencies with $(FRONTEND_PM)..."
> $(FRONTEND_INSTALL_CMD)

build-backend:
> set -euo pipefail
> echo "Building backend (compileall)..."
> $(BACKEND_BUILD_CMD)

build-frontend:
> set -euo pipefail
> echo "Building frontend..."
> $(FRONTEND_BUILD_CMD)

clean-frontend:
> set -euo pipefail
> echo "Cleaning frontend artifacts..."
> rm -rf frontend/.next frontend/node_modules frontend/package-lock.json

start-backend: | $(PID_DIR)
> set -euo pipefail
> port_busy=""; \
> if command -v lsof >/dev/null 2>&1; then \
>   port_busy=$$(lsof -iTCP:$(BACKEND_PORT) -sTCP:LISTEN -Pn 2>/dev/null | sed '/^$$/d' || true); \
> fi; \
> if [ -z "$$port_busy" ] && command -v ss >/dev/null 2>&1; then \
>   port_busy=$$(ss -ltnp "sport = :$(BACKEND_PORT)" 2>/dev/null | tail -n +2 || true); \
> fi; \
> if [ -n "$$port_busy" ]; then \
>   echo "Port $(BACKEND_PORT) is already in use:"; \
>   echo "$$port_busy"; \
>   exit 1; \
> fi
> if [ -f $(PID_DIR)/backend.pid ]; then \
>   old=$$(cat $(PID_DIR)/backend.pid); \
>   if kill -0 $$old 2>/dev/null; then \
>     echo "Backend already running (pid $$old). Stop it first with: make stop"; \
>     exit 1; \
>   else \
>     rm -f $(PID_DIR)/backend.pid; \
>   fi; \
> fi
> echo "Starting backend on $(BACKEND_BIND_HOST):$(BACKEND_PORT)..."
> nohup setsid sh -c '$(BACKEND_RUN_CMD)' > $(PID_DIR)/backend.log 2>&1 &
> echo $$! > $(PID_DIR)/backend.pid
> echo "Backend PGID $$! (logs: $(PID_DIR)/backend.log)"

start-frontend: | $(PID_DIR)
> set -euo pipefail
> port_busy=""; \
> if command -v lsof >/dev/null 2>&1; then \
>   port_busy=$$(lsof -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN -Pn 2>/dev/null | sed '/^$$/d' || true); \
> fi; \
> if [ -z "$$port_busy" ] && command -v ss >/dev/null 2>&1; then \
>   port_busy=$$(ss -ltnp "sport = :$(FRONTEND_PORT)" 2>/dev/null | tail -n +2 || true); \
> fi; \
> if [ -n "$$port_busy" ]; then \
>   echo "Port $(FRONTEND_PORT) is already in use:"; \
>   echo "$$port_busy"; \
>   exit 1; \
> fi
> if [ -f $(PID_DIR)/frontend.pid ]; then \
>   old=$$(cat $(PID_DIR)/frontend.pid); \
>   if kill -0 $$old 2>/dev/null; then \
>     echo "Frontend already running (pid $$old). Stop it first with: make stop"; \
>     exit 1; \
>   else \
>     rm -f $(PID_DIR)/frontend.pid; \
>   fi; \
> fi
> # verify build artifacts exist before starting
> if [ ! -f frontend/.next/BUILD_ID ] || [ ! -f frontend/.next/server/webpack-runtime.js ] || [ ! -d frontend/.next/server/chunks ]; then \
>   echo "Frontend build artifacts missing or incomplete. Run: make build-frontend"; \
>   exit 1; \
> fi
> if ! ls frontend/.next/server/chunks/*.js >/dev/null 2>&1; then \
>   echo "Frontend build artifacts missing server chunks. Run: make build-frontend"; \
>   exit 1; \
> fi
> echo "Starting frontend on $(FRONTEND_BIND_HOST):$(FRONTEND_PORT)..."
> nohup setsid sh -c 'cd frontend && NODE_ENV=production $(FRONTEND_PM) run start -- --hostname $(FRONTEND_BIND_HOST) --port $(FRONTEND_PORT)' > $(PID_DIR)/frontend.log 2>&1 &
> echo $$! > $(PID_DIR)/frontend.pid
> echo "Frontend PGID $$! (logs: $(PID_DIR)/frontend.log)"

run-all dev: | $(PID_DIR)
> set -euo pipefail
> trap 'echo ""; echo "Stopping servers..."; $(MAKE) stop; exit 0' INT TERM
> $(MAKE) stop >/dev/null 2>&1 || true
> $(MAKE) install-backend install-frontend
> $(MAKE) build-backend build-frontend
> $(MAKE) start-backend
> $(MAKE) start-frontend
> echo "Waiting for services to become healthy..."
> if ! $(MAKE) health; then \
>   if [ "$${COOKBOOKAI_DEBUG_RUNALL:-0}" = "1" ]; then \
>     echo "--- lsof backend $(BACKEND_PORT) ---"; \
>     lsof -iTCP:$(BACKEND_PORT) -sTCP:LISTEN -Pn 2>/dev/null || true; \
>     echo "--- lsof frontend $(FRONTEND_PORT) ---"; \
>     lsof -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN -Pn 2>/dev/null || true; \
>   fi; \
>   echo "Health checks failed. Stopping servers..."; \
>   $(MAKE) stop; \
>   exit 1; \
> fi
> echo "Backend: http://$(BACKEND_HOST):$(BACKEND_PORT)"
> echo "Frontend: http://$(FRONTEND_HOST):$(FRONTEND_PORT)"
> echo "Stop: make stop"
> back_pid=$$(cat $(PID_DIR)/backend.pid); front_pid=$$(cat $(PID_DIR)/frontend.pid)
> echo "Servers running (backend $$back_pid, frontend $$front_pid). Press Ctrl-C to stop."
> while kill -0 $$back_pid >/dev/null 2>&1 || kill -0 $$front_pid >/dev/null 2>&1; do \
>   sleep 2; \
> done

health:
> set -euo pipefail
> backend_url="http://$(BACKEND_HOST):$(BACKEND_PORT)/api/parse/health"
> frontend_base="http://$(FRONTEND_HOST):$(FRONTEND_PORT)"
> frontend_health="$$frontend_base/api/health"
> frontend_demo="$$frontend_base/demo"
> backend_ok=0; frontend_ok=0; frontend_ok_url=""
> for i in $$(seq 1 60); do \
>   if curl -sf "$$backend_url" >/dev/null 2>&1; then backend_ok=1; break; fi; \
>   sleep 1; \
> done
> for i in $$(seq 1 60); do \
>   if curl -sf "$$frontend_health" >/dev/null 2>&1; then frontend_ok=1; frontend_ok_url="$$frontend_health"; break; fi; \
>   if curl -sf "$$frontend_demo" >/dev/null 2>&1; then frontend_ok=1; frontend_ok_url="$$frontend_demo"; break; fi; \
>   if curl -sf "$$frontend_base" >/dev/null 2>&1; then frontend_ok=1; frontend_ok_url="$$frontend_base"; break; fi; \
>   sleep 1; \
> done
> if [ $$backend_ok -eq 1 ]; then \
>   echo "✅ Backend OK ($$backend_url)"; \
> else \
>   echo "❌ Backend FAIL ($$backend_url)"; \
>   tail -n 20 $(PID_DIR)/backend.log 2>/dev/null || true; \
>   exit 1; \
> fi
> if [ $$frontend_ok -eq 1 ]; then \
>   echo "✅ Frontend OK ($$frontend_ok_url)"; \
> else \
>   echo "❌ Frontend FAIL (checked $$frontend_health, $$frontend_demo, $$frontend_base)"; \
>   tail -n 50 $(PID_DIR)/frontend.log 2>/dev/null || true; \
>   if [ -d frontend/.next/server/chunks ]; then ls -la frontend/.next/server/chunks | head; fi; \
>   if [ -d frontend/.next/static/chunks ]; then ls -la frontend/.next/static/chunks | head; fi; \
>   exit 1; \
> fi

status:
> set -euo pipefail
> backend_url="http://$(BACKEND_HOST):$(BACKEND_PORT)"
> frontend_url="http://$(FRONTEND_HOST):$(FRONTEND_PORT)"
> echo "Backend: $$backend_url"
> if [ -f $(PID_DIR)/backend.pid ]; then \
>   pid=$$(cat $(PID_DIR)/backend.pid); \
>   if kill -0 $$pid 2>/dev/null; then \
>     echo "  running (pid $$pid)"; \
>   else \
>     echo "  pid file present but process not running"; \
>   fi; \
> else \
>   echo "  not running (no PID file)"; \
> fi
> if command -v lsof >/dev/null 2>&1; then \
>   lsof -iTCP:$(BACKEND_PORT) -sTCP:LISTEN -Pn 2>/dev/null || echo "  port $(BACKEND_PORT) not listening (or lsof permissions)"; \
> else \
>   echo "  (install lsof to display listening ports)"; \
> fi
> if command -v ss >/dev/null 2>&1; then \
>   ss -ltnp "sport = :$(BACKEND_PORT)" 2>/dev/null | tail -n +2 || true; \
> fi
> echo ""
> echo "Frontend: $$frontend_url"
> if [ -f $(PID_DIR)/frontend.pid ]; then \
>   pid=$$(cat $(PID_DIR)/frontend.pid); \
>   if kill -0 $$pid 2>/dev/null; then \
>     echo "  running (pid $$pid)"; \
>   else \
>     echo "  pid file present but process not running"; \
>   fi; \
> else \
>   echo "  not running (no PID file)"; \
> fi
> if command -v lsof >/dev/null 2>&1; then \
>   lsof -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN -Pn 2>/dev/null || echo "  port $(FRONTEND_PORT) not listening (or lsof permissions)"; \
> else \
>   echo "  (install lsof to display listening ports)"; \
> fi
> if command -v ss >/dev/null 2>&1; then \
>   ss -ltnp "sport = :$(FRONTEND_PORT)" 2>/dev/null | tail -n +2 || true; \
> fi
> echo ""
> echo "Logs: $(PID_DIR)/backend.log, $(PID_DIR)/frontend.log"
> echo "Stop: make stop"

stop:
> set -euo pipefail
> for name in backend frontend; do \
>   pid_file="$(PID_DIR)/$$name.pid"; \
>   if [ -f "$$pid_file" ]; then \
>     pid=$$(cat "$$pid_file"); \
>     if kill -0 $$pid 2>/dev/null; then \
>       pgid=$$(ps -o pgid= -p $$pid 2>/dev/null | tr -d ' '); \
>       target=$${pgid:-$$pid}; \
>       echo "Stopping $$name process group (pgid $$target)..."; \
>       kill -TERM -$$target 2>/dev/null || true; \
>       sleep 1; \
>       kill -KILL -$$target 2>/dev/null || true; \
>     else \
>       echo "Removing stale PID for $$name"; \
>     fi; \
>     rm -f "$$pid_file"; \
>   else \
>     echo "$$name not running (no PID file)"; \
>   fi; \
> done
> # Also clear any stray listeners on the expected ports
> if command -v lsof >/dev/null 2>&1; then \
>   lsof -t -iTCP:$(BACKEND_PORT) -sTCP:LISTEN -Pn 2>/dev/null | xargs -r kill 2>/dev/null || true; \
>   lsof -t -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN -Pn 2>/dev/null | xargs -r kill 2>/dev/null || true; \
> fi
> fuser -k $(BACKEND_PORT)/tcp 2>/dev/null || true; \
> fuser -k $(FRONTEND_PORT)/tcp 2>/dev/null || true; \
> if command -v ss >/dev/null 2>&1; then \
>   for port in $(BACKEND_PORT) $(FRONTEND_PORT); do \
>     ss -ltnp "sport = :$$port" 2>/dev/null | tail -n +2 | awk -F'pid=' '{for(i=2;i<=NF;i++){split($$i,a,/[,)]/); if(a[1]!=""){print a[1];}}}' | xargs -r kill 2>/dev/null || true; \
>   done; \
> fi

option2:
> $(PY) scripts/run_option2_pipeline.py --config configs/option2_pipeline.yaml

detect:
> $(PY) scripts/run_option2_pipeline.py --config configs/option2_pipeline.yaml --stop_after detect

weaklabel:
> $(PY) scripts/run_option2_pipeline.py --config configs/option2_pipeline.yaml --stop_after weaklabel

build-datasets:
> $(PY) scripts/run_option2_pipeline.py --config configs/option2_pipeline.yaml --stop_after build_datasets

sanity:
> $(PY) scripts/run_option2_pipeline.py --config configs/option2_pipeline.yaml --start_after build_datasets --stop_after sanity

phase4-run:
> $(PY) scripts/run_phase4_experiments.py --config configs/phase4_experiment.yaml

regen-highconf:
> $(PY) scripts/regenerate_highconf.py --overwrite

demo:
> $(MAKE) demo-stop
> # Ensure samples exist
> $(PY) scripts/generate_sample_assets.py --overwrite
> tmux new-session -d -s cookbookai-backend "cd backend && source ../.venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port $(BACKEND_PORT)"
> for i in $$(seq 1 30); do curl -sf http://localhost:$(BACKEND_PORT)/api/parse/health >/dev/null && break; sleep 1; done
> tmux new-session -d -s cookbookai-frontend "cd frontend && npm run dev -- --port $(FRONTEND_PORT)"
> for i in $$(seq 1 20); do curl -sf http://localhost:$(FRONTEND_PORT) >/dev/null && break; sleep 1; done
> powershell.exe Start-Process http://localhost:$(FRONTEND_PORT)/demo 2>/dev/null || echo "Open http://localhost:$(FRONTEND_PORT)/demo"

demo-stop:
> - tmux has-session -t cookbookai-backend 2>/dev/null && tmux kill-session -t cookbookai-backend || true
> - tmux has-session -t cookbookai-frontend 2>/dev/null && tmux kill-session -t cookbookai-frontend || true
> - lsof -ti:$(BACKEND_PORT) | xargs kill -9 2>/dev/null || true
> - lsof -ti:$(FRONTEND_PORT) | xargs kill -9 2>/dev/null || true

render-pages:
> set -euo pipefail
> echo "Rendering Boston PDF to PNGs (full book). Limit with COOKBOOKAI_MAX_PAGES or --max-pages)"
> python scripts/render_boston_pages.py --pdf data/raw/boston-cooking-school-1918.pdf --out-dir data/pages/boston --dpi 200 $(if $(COOKBOOKAI_MAX_PAGES),--max-pages $(COOKBOOKAI_MAX_PAGES),)

generate-curated-labels:
> set -euo pipefail
> echo "Generating token-level weak labels from curated recipe JSON files..."
> python scripts/generate_curated_weak_labels.py --curated-dir frontend/public/recipes/boston --ocr-jsonl data/ocr/boston_pages.jsonl --out data/labels/boston_curated_weak_labeled.jsonl --match-threshold 0.7 --min-coverage 0.5

build-dataset: generate-curated-labels
> set -euo pipefail
> echo "Building HF dataset from OCR + weak labels (merges: gold > corrections > curated > weak)"
> python scripts/build_boston_dataset.py --pdf data/raw/boston-cooking-school-1918.pdf --pages-dir data/pages/boston --ocr-jsonl data/ocr/boston_pages.jsonl --out-dir data/datasets/boston_layoutlmv3_full $(if $(COOKBOOKAI_MAX_PAGES),--max-pages $(COOKBOOKAI_MAX_PAGES),)

validate-dataset:
> set -euo pipefail
> python scripts/audit_dataset_labels.py --split train --dataset-dir data/datasets/boston_layoutlmv3_full/dataset_dict --fail-on-invalid --min-non-o-ratio 0.02

rebuild-dataset:
> set -euo pipefail
> rm -rf data/datasets/boston_layoutlmv3_full/dataset_dict
> python scripts/build_boston_dataset.py --pdf data/raw/boston-cooking-school-1918.pdf --pages-dir data/pages/boston --ocr-jsonl data/ocr/boston_pages.jsonl --out-dir data/datasets/boston_layoutlmv3_full --overwrite $(if $(COOKBOOKAI_MAX_PAGES),--max-pages $(COOKBOOKAI_MAX_PAGES),)
> $(MAKE) validate-dataset

build-recipe-only:
> set -euo pipefail
> echo "Building recipe-only dataset (dense supervision)..."
> python scripts/build_boston_dataset.py --pdf data/raw/boston-cooking-school-1918.pdf --pages-dir data/pages/boston --ocr-jsonl data/ocr/boston_pages.jsonl --out-dir data/datasets/boston_layoutlmv3_full --build-recipe-only $(if $(COOKBOOKAI_MAX_PAGES),--max-pages $(COOKBOOKAI_MAX_PAGES),)

rebuild-data: render-pages generate-curated-labels build-dataset
> @echo "Full data rebuild complete (with curated labels)."

demo-status:
> @echo "tmux sessions:"; tmux ls 2>/dev/null || echo "no tmux sessions"
> @echo "\nBackend health:"; if curl -sf http://localhost:$(BACKEND_PORT)/api/parse/health -o /tmp/cb_health.json; then (jq . /tmp/cb_health.json 2>/dev/null || python -m json.tool /tmp/cb_health.json 2>/dev/null || cat /tmp/cb_health.json); else echo "Backend not reachable"; fi
> @echo "\nFrontend status:"; if curl -sf http://localhost:$(FRONTEND_PORT) >/dev/null; then echo "Frontend reachable at http://localhost:$(FRONTEND_PORT)"; else echo "Frontend not reachable"; fi

frontend-diagnose:
> set -euo pipefail
> if [ -f frontend/.next/BUILD_ID ]; then echo "BUILD_ID: $$(cat frontend/.next/BUILD_ID)"; else echo "BUILD_ID missing"; fi
> if [ -d frontend/.next/server/chunks ]; then echo ".next/server/chunks:"; ls -la frontend/.next/server/chunks | head; else echo ".next/server/chunks missing"; fi
> if [ -d frontend/.next/static/chunks ]; then echo ".next/static/chunks:"; ls -la frontend/.next/static/chunks | head; else echo ".next/static/chunks missing"; fi

ensure-validation-split:
> set -euo pipefail
> echo "Ensuring dataset has non-empty validation split..."
> python scripts/ensure_validation_split.py --dataset-dir data/datasets/boston_layoutlmv3_full --val-ratio 0.1 --seed 42

audit-dataset:
> set -euo pipefail
> mkdir -p docs/debug
> echo "Auditing dataset label distribution..."
> python scripts/audit_dataset_labels.py --split train > docs/debug/dataset_label_audit.txt
> echo "Wrote docs/debug/dataset_label_audit.txt"

sanity-labels:
> set -euo pipefail
> echo "Running label sanity checks on encoded dataset..."
> python scripts/sanity_check_encoded_labels.py \
>   --dataset-dir data/datasets/boston_layoutlmv3_full/dataset_dict \
>   --num-samples 20 \
>   --splits train validation

rebuild-recipe-only-250p:
> set -euo pipefail
> echo "Rebuilding recipe-only dataset (250 pages, balanced ingredients+instructions)..."
> echo "1. Rebuilding merged JSONL with heuristic labels..."
> python scripts/build_boston_dataset.py \
>   --pdf data/raw/boston-cooking-school-1918.pdf \
>   --pages-dir data/pages/boston \
>   --ocr-jsonl data/ocr/boston_pages.jsonl \
>   --out-dir data/datasets/boston_layoutlmv3_full \
>   --build-recipe-only \
>   --overwrite
> echo ""
> echo "2. Running sanity checks on recipe-only dataset..."
> python scripts/sanity_check_encoded_labels.py \
>   --dataset-dir data/datasets/boston_layoutlmv3_recipe_only/dataset_dict \
>   --num-samples 20 \
>   --splits train validation
> echo ""
> echo "3. Printing ingredient coverage stats..."
> python -c "import json; stats = json.load(open('data/datasets/boston_layoutlmv3_recipe_only/stats.json')); dist = stats.get('label_distribution', {}); ing = dist.get('INGREDIENT_LINE', {}); inst = dist.get('INSTRUCTION_STEP', {}); print(f\"INGREDIENT_LINE: {ing.get('count', 0)} tokens ({ing.get('percent', 0)*100:.2f}%)\"); print(f\"INSTRUCTION_STEP: {inst.get('count', 0)} tokens ({inst.get('percent', 0)*100:.2f}%)\"); print(f\"Total pages: {stats.get('total_pages', 0)}\")"
> echo ""
> echo "✅ Recipe-only dataset rebuilt at data/datasets/boston_layoutlmv3_recipe_only/"
smoke:
> $(PY) scripts/verify_sample_assets.py || ($(PY) scripts/generate_sample_assets.py --overwrite && $(PY) scripts/verify_sample_assets.py)
> $(PY) -m scripts.smoke.smoke_backend
> $(PY) -m scripts.smoke.smoke_upload

smoke-demo:
> $(PY) backend/scripts/smoke_demo.py

hello:
> @echo "Makefile works ✅"

fix-line-endings:
> dos2unix Makefile || true

backend:
> $(MAKE) backend-stop
> @bash ./scripts/dev_backend.sh

backend-health:
> if curl -sf http://localhost:$(BACKEND_PORT)/api/parse/health -o /tmp/cb_health.json; then \
>   (jq . /tmp/cb_health.json 2>/dev/null || python -m json.tool /tmp/cb_health.json 2>/dev/null || cat /tmp/cb_health.json); \
> else \
>   echo "Backend not reachable at http://localhost:$(BACKEND_PORT)/api/parse/health. Start it with: make backend"; \
> fi

backend-stop:
> - tmux has-session -t cookbookai-backend 2>/dev/null && tmux kill-session -t cookbookai-backend || true
> @fuser -k $(BACKEND_PORT)/tcp 2>/dev/null || true
> @echo "Stopped backend on port $(BACKEND_PORT) (and tmux session if present)"

check-makefile:
> @python - << 'PY'
> import pathlib, re, sys
> text = pathlib.Path("Makefile").read_text()
> if "\r\n" in text: print("ERROR: Makefile contains CRLF line endings. Run: make fix-line-endings"); sys.exit(1)
> bad = [(i, line[:50]) for i, line in enumerate(text.splitlines(), start=1) if re.match(r"^ {2,}[@a-zA-Z0-9_./-]", line)]
> if bad: print("ERROR: Recipe lines appear to be indented with spaces."); [print(f"  line {i}: {s!r}") for i, s in bad[:10]]; print("Use .RECIPEPREFIX := > and prefix recipe lines with '>'"); sys.exit(1)
> print("Makefile OK ✅")
> PY
