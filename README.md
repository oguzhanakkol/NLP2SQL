# Natural Language to SQL Pipeline

A comprehensive academic research pipeline for converting natural language questions to SQL queries using a three-phase approach with state-of-the-art models and evaluation on the BIRD benchmark.

## Overview

This pipeline implements a sophisticated Natural Language to SQL conversion system designed for academic research. It consists of three main phases:

1. **Phase 1: Schema Linking & Pruning** - Uses Hybrid Table-Column Schema Linking (HyTCSL) with embedding models and LLM refinement
2. **Phase 2: SQL Candidate Generation** - Multi-model ensemble approach with various schema representations and stochastic sampling
3. **Phase 3: SQL Selection** - Advanced candidate selection using LLM critics, voting mechanisms, and multi-criteria ranking

## Key Features

- **Multi-Model Support**: Both open-source (XiYanSQL, Arctic) and commercial models (GPT-4o, Gemini)
- **Advanced Schema Linking**: HyTCSL approach with embedding-based similarity and LLM refinement
- **Multiple Schema Representations**: M-Schema, DDL, JSON, and Markdown formats
- **Robust Evaluation**: BIRD benchmark evaluation with execution accuracy metrics
- **Checkpoint Management**: Resume long-running sessions (important for H100 cluster constraints)
- **Comprehensive Logging**: Detailed JSON logs, prompt saving, and statistics tracking
- **Cost Tracking**: Monitor token usage and API costs for all commercial models
- **Model Output Analysis**: Complete capture and analysis of all model interactions
- **Progressive Evaluation**: Real-time metrics during pipeline execution

## Project Structure

```
nlp2sql-pipeline/
├── src/                                    # Source code
│   ├── __init__.py
│   ├── core/                              # Core functionality
│   │   ├── __init__.py
│   │   ├── config_manager.py              # Configuration management
│   │   ├── logger.py                      # Logging system
│   │   ├── data_loader.py                 # BIRD data loading
│   │   ├── model_manager.py               # Model management
│   │   ├── statistics_tracker.py          # Statistics and metrics
│   │   └── checkpoint_manager.py          # Checkpoint handling
│   └── phases/                            # Pipeline phases
│       ├── phase1_schema_linking/         # Schema linking implementation
│       │   ├── __init__.py
│       │   └── schema_linker.py           # HyTCSL + LLM refinement
│       ├── phase2_sql_generation/         # SQL candidate generation
│       │   ├── __init__.py
│       │   └── sql_generator.py           # Multi-model SQL generation
│       └── phase3_sql_selection/          # SQL selection and ranking
│           ├── __init__.py
│           └── sql_selector.py            # LLM critic + voting
├── tools/                                 # Utility tools
│   ├── __init__.py
│   ├── schema_representations/            # Schema format tools
│   │   ├── __init__.py
│   │   └── mschema.py                    # M-Schema implementation
│   ├── evaluation/                       # Evaluation tools
│   │   ├── __init__.py
│   │   └── bird_evaluator.py             # BIRD benchmark evaluator
│   └── model_output_analyzer.py          # Model output analysis tool
├── tests/                                # Test scripts
│   ├── __init__.py
│   ├── test_core.py                      # Core components tests
│   ├── test_evaluation.py               # Evaluation tests
│   ├── test_full_pipeline.py            # Full pipeline tests
│   ├── test_phase1.py                   # Phase 1 tests
│   ├── test_phase2.py                   # Phase 2 tests
│   └── test_phase3.py                   # Phase 3 tests
├── configs/                             # Configuration files
│   └── pipeline_config.yaml            # Main configuration file
├── data/                               # Data directory (created by setup)
│   ├── bird_benchmark/                # BIRD dataset (user provided)
│   │   ├── dev.json                   # Questions and metadata
│   │   ├── dev_gold.sql              # Ground truth SQL (optional)
│   │   └── dev_databases/            # SQLite database files
│   ├── checkpoints/                  # Checkpoint storage (auto-created)
│   ├── results/                     # Pipeline results (auto-created)
│   ├── candidate_pools/            # SQL candidate pools (auto-created)
│   ├── models/                    # Local model cache (auto-created)
│   └── cache/                    # General cache (auto-created)
├── logs/                        # Log files (auto-created)
│   ├── prompts/                # Saved prompts for inspection
│   └── *.log                  # Various log files
├── main.py                   # Main pipeline script
├── requirements.txt         # Python dependencies
├── Makefile               # Build and test commands
├── .gitignore           # Git ignore patterns
└── README.md           # Project documentation
```

## Quick Start

### 1. Installation

**Automated Setup (Recommended)**:
```bash
# See all available commands
make help

# Complete setup (installs dependencies + creates directories)
make setup
```

**Manual Setup**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{bird_benchmark,checkpoints,results,models,cache} logs/prompts
```

### 2. Data Setup

**Download BIRD benchmark dataset** and place it in `data/bird_benchmark/` (if not already installed):
```
data/bird_benchmark/
├── dev.json                 # Questions and metadata
├── dev_gold.sql            # Ground truth SQL (optional)
└── dev_databases/          # SQLite database files
    ├── california_schools/
    ├── card_games/
    └── ...
```

**Verify data structure**:
```bash
make validate-data
```

### 3. API Keys Setup

For commercial models (GPT-4o, Gemini):
```bash
export OPENAI_API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

### 4. Environment Validation

Check your setup:
```bash
make health-check
```

## Usage

### Demo Mode (Recommended First Step)

Test the pipeline with a single question:
```bash
make demo                    # Question 0
make demo-q5                # Question 5
```

Or using Python directly:
```bash
python main.py --demo --demo-question-id 0
```

### Full Pipeline Execution

```bash
# Run complete pipeline
make run

# Process subset (100 questions)
make run-subset

# Resume from checkpoint
make run-resume

# Quick test (5 questions)
make run-test
```

### Advanced Options

```bash
# Custom configuration
python main.py --config configs/custom_config.yaml

# Specific number of questions
python main.py --max-questions 100

# Resume with custom limit
python main.py --max-questions 500 --resume
```

## Testing

### Comprehensive Test Suite

```bash
make test                   # Run all tests
make test-core             # Core components
make test-phase1           # Schema linking
make test-phase2           # SQL generation  
make test-phase3           # SQL selection
make test-evaluation       # BIRD evaluation
```

### Individual Component Tests

```bash
make test-data             # Validate data loading
make test-models           # Check model configurations
make validate-config       # Validate pipeline configuration
make validate-env          # Check environment setup
```

## Configuration

The pipeline is configured through `configs/pipeline_config.yaml`. Key sections include:

### Cost Optimization

**Local Models Only (Zero API Cost)**:
```yaml
phase1_schema_linking:
  refinement_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
  refinement_model_type: "local"

phase3_sql_selection:
  ranking_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
  ranking_model_type: "local"
```

**Commercial Models (Better Performance)**:
```yaml
phase1_schema_linking:
  refinement_model: "gpt4o"
  refinement_model_type: "commercial"

phase3_sql_selection:
  ranking_model: "gpt4o"
  ranking_model_type: "commercial"
```

### Model Configuration
```yaml
models:
  embedding:
    model_name: "Qwen/Qwen3-Embedding-8B"
    device: "cuda"
  
  sql_generation:
    models:
      - name: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
        type: "open_source"
        device: "cuda"
        temperature_options: [0.1, 0.3, 0.7]
      - name: "Snowflake/Arctic-Text2SQL-R1-7B"
        type: "open_source"
        device: "cuda"
        temperature_options: [0.1, 0.3, 0.7]
  
  commercial:
    gpt4o:
      api_key_env: "OPENAI_API_KEY"
      model: "gpt-4o"
      max_tokens: 1000
      temperature: 0.1
    gemini:
      api_key_env: "GEMINI_API_KEY"
      model: "gemini-1.5-pro"
      max_tokens: 1000
      temperature: 0.1
```

### Phase Configuration
```yaml
phase1_schema_linking:
  top_k_tables: 10
  top_k_columns_per_table: 5
  enable_llm_refinement: true
  include_database_descriptions: true

phase2_sql_generation:
  candidates_per_model: 1
  schema_representations: ["m_schema", "ddl", "json", "markdown"]
  temperature_values: [0.1, 0.3, 0.7]
  enable_checkpointing: true
  enable_sql_fixing: true

phase3_sql_selection:
  remove_syntax_errors: true
  remove_duplicates: true
  enable_majority_voting: true
  enable_value_alignment: true
  enable_self_consistency: true
```

### Logging and Output Configuration
```yaml
logging:
  log_level: "INFO"
  save_prompts: true
  log_model_outputs: true
  json_logs: true

execution:
  show_progressive_metrics: true
  show_sql_execution_results: true
  include_model_outputs_in_results: true
```

## Methodology

### Phase 1: Schema Linking & Pruning (HyTCSL + M-Schema)

1. **Embedding-based Retrieval**: Uses Qwen3-Embedding to find semantically similar tables and columns
2. **Content/Value Linking**: Scans database for question-referenced values
3. **Schema Expansion**: Includes necessary tables for joins
4. **LLM Refinement**: Uses GPT-4o/Gemini/local models to refine the selection
5. **M-Schema Construction**: Creates structured schema representation with examples

### Phase 2: SQL Candidate Generation (Multi-Variant Ensemble)

1. **Multiple Schema Representations**: M-Schema, DDL, JSON, and Markdown formats
2. **Model Ensemble**: Open-source models (XiYanSQL, Arctic)
3. **Stochastic Sampling**: Multiple temperatures (0.1, 0.3, 0.7) for diversity
4. **Few-shot Prompting**: Contextual examples for better generation
5. **SQL Validation & Fixing**: Syntax checking and automatic correction with LLMs

### Phase 3: SQL Selection (Ranking & Voting)

1. **Validity Filtering**: Remove syntax errors and duplicates using SQLGlot
2. **LLM Critic**: Chain-of-thought ranking with GPT-4o/Gemini/local models
3. **Value Alignment**: Ensure question values appear in SQL
4. **Self-Consistency**: Boost candidates generated by multiple models
5. **Multi-Criteria Ranking**: Weighted combination of all scores

## Evaluation

The pipeline uses BIRD benchmark evaluation with execution accuracy:

- **Overall Accuracy**: Percentage of queries producing correct results
- **Execution Success Rate**: Percentage of queries that execute without errors
- **Difficulty Breakdown**: Performance by question difficulty (simple/moderate/challenging)
- **Database Breakdown**: Performance by database
- **Progressive Metrics**: Real-time accuracy tracking during execution

## Model Output Analysis

The pipeline captures all model interactions for detailed analysis:

```bash
# Analyze model outputs from logs
make analyze-logs

# Use the model output analyzer tool
python tools/model_output_analyzer.py --log-file logs/pipeline_execution.json --question-id 0

# Export detailed model interactions
python tools/model_output_analyzer.py --log-file logs/pipeline_execution.json --question-id 0 --save-outputs model_analysis.json
```

## Monitoring and Logging

The pipeline provides comprehensive logging:

- **Console Output**: Real-time progress with colored output and progress bars
- **JSON Logs**: Structured logs for analysis (`logs/pipeline_execution.json`)
- **Phase Logs**: Separate logs for each phase
- **Prompt Logs**: All prompts saved for inspection (`logs/prompts/`)
- **Model Output Logs**: Complete capture of all model interactions
- **Progressive Metrics**: Real-time accuracy and execution statistics
- **Cost Tracking**: Detailed API usage and cost monitoring

## Checkpointing

For long-running sessions (important for H100 clusters):

- **Automatic Checkpointing**: Saves progress every N questions (configurable)
- **Resume Capability**: Continue from last checkpoint with `make run-resume`
- **Configuration Tracking**: Ensures consistency across sessions
- **Candidate Pool Persistence**: Saves generated candidates to avoid regeneration
- **Session Management**: Automatic checkpoint on timeout for cluster environments

## Performance Considerations

- **Memory Management**: Automatic model cache clearing and GPU memory monitoring
- **GPU Optimization**: Efficient model loading and intelligent model reuse
- **Parallel Processing**: Multi-CPU evaluation (configurable)
- **Checkpoint Strategy**: Balance between safety and performance
- **Cost Optimization**: Support for local models to eliminate API costs

## Cost Management

- **Token Tracking**: Monitor input/output tokens for all models
- **Cost Estimation**: Real-time API cost tracking for commercial models
- **Usage Reports**: Export detailed usage statistics
- **Budget Controls**: Configure model types to control costs
- **Local Model Support**: Use local models exclusively for zero API costs

## Academic Research Features

- **Reproducibility**: Fixed random seeds and deterministic modes
- **Ablation Studies**: Easy component enabling/disabling via configuration
- **Detailed Metrics**: Comprehensive evaluation and statistical analysis
- **Export Capabilities**: Results in academic-friendly JSON formats
- **Model Interaction Capture**: Complete logging of all model prompts and outputs
- **Progressive Analysis**: Real-time performance monitoring during execution

## Maintenance

```bash
# Clean temporary files
make clean

# Clean all logs and caches
make clean-all

# Update dependencies
make update-deps

# Check project statistics
make show-stats

# View disk usage
make disk-usage
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   ```bash
   # Check CUDA availability
   make validate-env
   
   # Clear model cache if corrupted
   make clean-cache
   ```

2. **API Rate Limits**:
   - Use local models instead: Set `refinement_model_type: "local"` and `ranking_model_type: "local"`
   - Reduce `candidates_per_model` in configuration
   - Increase delays between API calls

3. **Memory Issues**:
   - Reduce `max_questions` for testing
   - Enable model cache clearing in configuration
   - Use smaller models or reduce batch sizes

4. **Database Access**:
   ```bash
   # Verify BIRD dataset structure
   make validate-data
   
   # Check database permissions
   ls -la data/bird_benchmark/dev_databases/
   ```

5. **Checkpoint Issues**:
   ```bash
   # Clear corrupted checkpoints
   make clean-checkpoints
   
   # Resume without checkpoint
   python main.py  # (don't use --resume)
   ```

### Debug Mode

Enable verbose logging:
```yaml
logging:
  log_level: "DEBUG"
  save_prompts: true
  log_model_outputs: true
```

### Performance Debugging

```bash
# Run performance benchmark
make benchmark

# Monitor resource usage
make validate-env

# Analyze execution logs
make analyze-logs
```

## Contact

For questions or collaboration opportunities, please contact [oraakkol674@gmail.com].

---

*This pipeline is designed for academic research and educational purposes. It implements state-of-the-art methods for natural language to SQL conversion with comprehensive evaluation on the BIRD benchmark.*