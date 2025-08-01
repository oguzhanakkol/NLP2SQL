.PHONY: help install install-dev setup test test-core test-data test-models test-phase1 test-phase2 test-phase3 test-evaluation test-full-pipeline
.PHONY: demo run run-resume run-subset clean clean-logs clean-cache clean-checkpoints clean-all
.PHONY: lint format check-format validate-config validate-data
.PHONY: docs docker-build docker-run
.PHONY: export-results analyze-logs benchmark

help:
	@echo "Natural Language to SQL Pipeline - Available Commands"
	@echo "====================================================="
	@echo ""
	@echo "🚀 SETUP & INSTALLATION:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup            Complete setup (install + create directories)"
	@echo ""
	@echo "🧪 TESTING:"
	@echo "  test             Run all tests"
	@echo "  test-core        Test core components"
	@echo "  test-data        Test data loading"
	@echo "  test-models      Test model management"
	@echo "  test-phase1      Test Phase 1 (Schema Linking)"
	@echo "  test-phase2      Test Phase 2 (SQL Generation)"
	@echo "  test-phase3      Test Phase 3 (SQL Selection)"
	@echo "  test-evaluation  Test BIRD evaluation"
	@echo "  test-full        Test complete pipeline"
	@echo ""
	@echo "▶️  EXECUTION:"
	@echo "  demo             Run demo with single question"
	@echo "  run              Run full pipeline"
	@echo "  run-resume       Resume from checkpoint"
	@echo "  run-subset       Run on subset (100 questions)"
	@echo ""
	@echo "🧹 MAINTENANCE:"
	@echo "  clean            Clean temporary files"
	@echo "  clean-logs       Clean log files"
	@echo "  clean-cache      Clean model cache"
	@echo "  clean-checkpoints Clean checkpoint files"
	@echo "  clean-all        Clean everything"
	@echo ""
	@echo "📋 CODE QUALITY:"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black"
	@echo "  check-format     Check code formatting"
	@echo ""
	@echo "✅ VALIDATION:"
	@echo "  validate-config  Validate configuration"
	@echo "  validate-data    Validate BIRD dataset"
	@echo ""
	@echo "📊 ANALYSIS:"
	@echo "  analyze-logs     Analyze execution logs"
	@echo "  export-results   Export latest results"
	@echo "  benchmark        Run performance benchmark"
	@echo ""

PYTHON := python3
PIP := pip3
SRC_DIR := src
TEST_DIR := tests
CONFIG_FILE := configs/pipeline_config.yaml
DATA_DIR := data
LOGS_DIR := logs
RESULTS_DIR := data/results

RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

install:
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

install-dev: install
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install pytest pytest-cov black flake8 mypy
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

setup: install
	@echo "$(BLUE)Setting up project directories...$(NC)"
	mkdir -p $(DATA_DIR)/bird_benchmark
	mkdir -p $(DATA_DIR)/checkpoints
	mkdir -p $(DATA_DIR)/results
	mkdir -p $(DATA_DIR)/models
	mkdir -p $(DATA_DIR)/cache
	mkdir -p $(LOGS_DIR)/prompts
	@echo "$(GREEN)✓ Project setup complete$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "1. Place BIRD dataset in $(DATA_DIR)/bird_benchmark/"
	@echo "2. Set API keys: export OPENAI_API_KEY=... export GEMINI_API_KEY=..."
	@echo "3. Run: make demo"

# ============================================================================
# TESTING
# ============================================================================

test: test-core test-data test-phase1 test-phase2 test-phase3 test-evaluation
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-core:
	@echo "$(BLUE)Testing core components...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) test_core.py

test-data:
	@echo "$(BLUE)Testing data loading...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) -c "import sys; sys.path.append('../src'); from src.core.data_loader import BirdDataLoader; from src.core.config_manager import ConfigManager; loader = BirdDataLoader(ConfigManager()); stats = loader.get_dataset_statistics(); print(f'Found {stats[\"total_questions\"]} questions, {stats[\"total_databases\"]} databases')"

test-models:
	@echo "$(BLUE)Testing model management...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) -c "import sys; sys.path.append('../src'); from src.core.model_manager import ModelManager; from src.core.config_manager import ConfigManager; mm = ModelManager(ConfigManager()); print('Model manager initialized successfully')"

test-phase1:
	@echo "$(BLUE)Testing Phase 1 (Schema Linking)...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) test_phase1.py

test-phase2:
	@echo "$(BLUE)Testing Phase 2 (SQL Generation)...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) test_phase2.py

test-phase3:
	@echo "$(BLUE)Testing Phase 3 (SQL Selection)...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) test_phase3.py

test-evaluation:
	@echo "$(BLUE)Testing BIRD evaluation...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) test_evaluation.py

test-full:
	@echo "$(BLUE)Testing complete pipeline...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) test_full_pipeline.py

test-pytest:
	@if command -v pytest >/dev/null 2>&1; then \
		echo "$(BLUE)Running tests with pytest...$(NC)"; \
		pytest $(TEST_DIR)/ -v --tb=short; \
	else \
		echo "$(RED)pytest not installed. Run: make install-dev$(NC)"; \
		exit 1; \
	fi

test-coverage:
	@if command -v pytest >/dev/null 2>&1; then \
		echo "$(BLUE)Running tests with coverage...$(NC)"; \
		pytest $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term; \
	else \
		echo "$(RED)pytest not installed. Run: make install-dev$(NC)"; \
		exit 1; \
	fi

# ============================================================================
# EXECUTION
# ============================================================================

demo:
	@echo "$(BLUE)Running demo mode...$(NC)"
	$(PYTHON) main.py --demo --demo-question-id 0

demo-q%:
	@echo "$(BLUE)Running demo with question $*...$(NC)"
	$(PYTHON) main.py --demo --demo-question-id $*

run:
	@echo "$(BLUE)Running full pipeline...$(NC)"
	$(PYTHON) main.py

run-resume:
	@echo "$(BLUE)Resuming pipeline from checkpoint...$(NC)"
	$(PYTHON) main.py --resume

run-subset:
	@echo "$(BLUE)Running pipeline on subset (100 questions)...$(NC)"
	$(PYTHON) main.py --max-questions 100

run-config:
	@if [ -z "$(CONFIG)" ]; then \
		echo "$(RED)Usage: make run-config CONFIG=path/to/config.yaml$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running pipeline with custom config: $(CONFIG)$(NC)"
	$(PYTHON) main.py --config $(CONFIG)

run-test:
	@echo "$(BLUE)Running pipeline test (5 questions)...$(NC)"
	$(PYTHON) main.py --max-questions 5

# ============================================================================
# MAINTENANCE & CLEANING
# ============================================================================

clean:
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Temporary files cleaned$(NC)"

clean-logs:
	@echo "$(BLUE)Cleaning log files...$(NC)"
	rm -rf $(LOGS_DIR)/*.log
	rm -rf $(LOGS_DIR)/*.json
	rm -rf $(LOGS_DIR)/prompts/*
	@echo "$(GREEN)✓ Log files cleaned$(NC)"

clean-cache:
	@echo "$(BLUE)Cleaning model cache...$(NC)"
	rm -rf $(DATA_DIR)/models/*
	rm -rf $(DATA_DIR)/cache/*
	@echo "$(GREEN)✓ Model cache cleaned$(NC)"

clean-checkpoints:
	@echo "$(BLUE)Cleaning checkpoint files...$(NC)"
	rm -rf $(DATA_DIR)/checkpoints/*
	@echo "$(GREEN)✓ Checkpoint files cleaned$(NC)"

clean-results:
	@echo "$(BLUE)Cleaning result files...$(NC)"
	rm -rf $(RESULTS_DIR)/*
	@echo "$(GREEN)✓ Result files cleaned$(NC)"

clean-all: clean clean-logs clean-cache clean-checkpoints clean-results
	@echo "$(GREEN)✓ All cleaning completed$(NC)"

# ============================================================================
# CODE QUALITY
# ============================================================================

lint:
	@echo "$(BLUE)Running linting checks...$(NC)"
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=120 --ignore=E203,W503; \
		echo "$(GREEN)✓ Linting completed$(NC)"; \
	else \
		echo "$(RED)flake8 not installed. Run: make install-dev$(NC)"; \
	fi

format:
	@echo "$(BLUE)Formatting code with black...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black $(SRC_DIR) $(TEST_DIR) --line-length=120; \
		echo "$(GREEN)✓ Code formatting completed$(NC)"; \
	else \
		echo "$(RED)black not installed. Run: make install-dev$(NC)"; \
	fi

check-format:
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black $(SRC_DIR) $(TEST_DIR) --check --line-length=120; \
		echo "$(GREEN)✓ Code formatting is correct$(NC)"; \
	else \
		echo "$(RED)black not installed. Run: make install-dev$(NC)"; \
	fi

type-check:
	@echo "$(BLUE)Running type checks...$(NC)"
	@if command -v mypy >/dev/null 2>&1; then \
		mypy $(SRC_DIR) --ignore-missing-imports; \
		echo "$(GREEN)✓ Type checking completed$(NC)"; \
	else \
		echo "$(RED)mypy not installed. Run: make install-dev$(NC)"; \
	fi

# ============================================================================
# VALIDATION
# ============================================================================

validate-config:
	@echo "$(BLUE)Validating configuration...$(NC)"
	$(PYTHON) -c "from src.core.config_manager import ConfigManager; cm = ConfigManager('$(CONFIG_FILE)'); print('✓ Configuration is valid')"

validate-data:
	@echo "$(BLUE)Validating BIRD dataset...$(NC)"
	$(PYTHON) -c "from src.core.data_loader import BirdDataLoader; from src.core.config_manager import ConfigManager; loader = BirdDataLoader(ConfigManager()); result = loader.validate_dataset(); print('✓ Dataset is valid' if result['valid'] else '✗ Dataset validation failed'); [print(f'Error: {e}') for e in result['errors']]"

validate-models:
	@echo "$(BLUE)Validating model configurations...$(NC)"
	$(PYTHON) -c "from src.core.model_manager import ModelManager; from src.core.config_manager import ConfigManager; mm = ModelManager(ConfigManager()); models = mm.get_available_sql_models(); print(f'✓ Found {len(models)} SQL models configured')"

validate-env:
	@echo "$(BLUE)Validating environment...$(NC)"
	@$(PYTHON) -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
	@if [ -n "$$OPENAI_API_KEY" ]; then echo "✓ OpenAI API key set"; else echo "⚠ OpenAI API key not set"; fi
	@if [ -n "$$GEMINI_API_KEY" ]; then echo "✓ Gemini API key set"; else echo "⚠ Gemini API key not set"; fi

# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

analyze-logs:
	@echo "$(BLUE)Analyzing execution logs...$(NC)"
	@if [ -f "$(LOGS_DIR)/pipeline_execution.json" ]; then \
		$(PYTHON) tools/model_output_analyzer.py --log-file "$(LOGS_DIR)/pipeline_execution.json" --question-id 0 2>/dev/null || echo "$(YELLOW)No logs found for analysis$(NC)"; \
	else \
		echo "$(YELLOW)No execution logs found. Run the pipeline first.$(NC)"; \
	fi

export-results:
	@echo "$(BLUE)Exporting latest results...$(NC)"
	@LATEST_RESULT=$$(ls -t $(RESULTS_DIR)/pipeline_results_*.json 2>/dev/null | head -1); \
	if [ -n "$$LATEST_RESULT" ]; then \
		cp "$$LATEST_RESULT" "$(RESULTS_DIR)/latest_results.json"; \
		echo "$(GREEN)✓ Latest results exported to $(RESULTS_DIR)/latest_results.json$(NC)"; \
	else \
		echo "$(YELLOW)No results found. Run the pipeline first.$(NC)"; \
	fi

show-stats:
	@echo "$(BLUE)Showing project statistics...$(NC)"
	@echo "Lines of code:"
	@find $(SRC_DIR) -name "*.py" | xargs wc -l | tail -1
	@echo "Test files:"
	@find $(TEST_DIR) -name "*.py" | wc -l
	@echo "Configuration files:"
	@find configs -name "*.yaml" | wc -l
	@if [ -d "$(DATA_DIR)/bird_benchmark" ]; then \
		echo "BIRD dataset status: ✓ Present"; \
	else \
		echo "BIRD dataset status: ✗ Missing"; \
	fi

benchmark:
	@echo "$(BLUE)Running performance benchmark...$(NC)"
	@echo "Testing with 10 questions..."
	time $(PYTHON) main.py --max-questions 10

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs:
	@echo "$(BLUE)Generating documentation...$(NC)"
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs docs/_build; \
		echo "$(GREEN)✓ Documentation generated in docs/_build$(NC)"; \
	else \
		echo "$(RED)Sphinx not installed. Install with: pip install sphinx$(NC)"; \
	fi

docs-serve:
	@echo "$(BLUE)Serving documentation...$(NC)"
	@if [ -d "docs/_build" ]; then \
		cd docs/_build && $(PYTHON) -m http.server 8000; \
	else \
		echo "$(RED)Documentation not built. Run: make docs$(NC)"; \
	fi

# ============================================================================
# DOCKER (Optional)
# ============================================================================

docker-build:
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t nlp2sql-pipeline .

docker-run:
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		-e OPENAI_API_KEY \
		-e GEMINI_API_KEY \
		nlp2sql-pipeline

# ============================================================================
# UTILITY TARGETS
# ============================================================================

disk-usage:
	@echo "$(BLUE)Disk usage by directory:$(NC)"
	@du -sh $(DATA_DIR) $(LOGS_DIR) $(RESULTS_DIR) 2>/dev/null || echo "Directories not found"

check-deps:
	@echo "$(BLUE)Checking Python dependencies...$(NC)"
	$(PIP) check

update-deps:
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt

show-config:
	@echo "$(BLUE)Current configuration:$(NC)"
	@head -20 $(CONFIG_FILE)

create-sample-data:
	@echo "$(BLUE)Creating sample dataset for testing...$(NC)"
	mkdir -p $(DATA_DIR)/sample_bird
	@echo "Sample dataset created in $(DATA_DIR)/sample_bird"
	@echo "$(YELLOW)Note: This is just a placeholder. You need the real BIRD dataset.$(NC)"

health-check: validate-env validate-config
	@echo "$(GREEN)✓ Health check completed$(NC)"

install-hooks:
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"; \
	else \
		echo "$(YELLOW)pre-commit not installed. Install with: pip install pre-commit$(NC)"; \
	fi

help-%:
	@echo "Help for target '$*':"
	@grep -A 5 "^$*:" Makefile | grep "^#" | sed 's/^#\s*//'