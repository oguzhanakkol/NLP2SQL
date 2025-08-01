# Configuration Guide

This guide provides comprehensive documentation for all configuration options in the Natural Language to SQL Pipeline.

## Configuration File Structure

The main configuration file is `configs/pipeline_config.yaml` with the following sections:

```yaml
data:                          # Data paths and dataset configuration
logging:                       # Logging and monitoring settings  
models:                        # Model definitions and parameters
phase1_schema_linking:         # Phase 1 specific settings
phase2_sql_generation:         # Phase 2 specific settings
phase3_sql_selection:          # Phase 3 specific settings
evaluation:                    # Evaluation and metrics configuration
execution:                     # Pipeline execution parameters
advanced:                      # Advanced features and debugging
```

## Data Configuration

### Basic Data Paths

```yaml
data:
  # BIRD benchmark dataset paths
  bird_benchmark_path: "data/bird_benchmark"              # Root directory for BIRD benchmark
  dev_json_path: "data/bird_benchmark/dev.json"           # Development questions in JSON format
  databases_path: "data/bird_benchmark/dev_databases"     # SQLite database files directory
  
  # Pipeline output directories (created automatically)
  checkpoints_path: "data/checkpoints"                    # Checkpoint files for resuming execution
  results_path: "data/results"                           # Final pipeline results and evaluations
  candidate_pools_path: "data/candidate_pools"           # SQL candidate pools for reuse
```

**Usage Notes**:
- All paths are relative to the project root
- Directories are created automatically if they don't exist
- BIRD dataset must be downloaded separately

## Logging Configuration

### Basic Logging Settings

```yaml
logging:
  # Basic logging settings
  log_directory: "logs"                                   # Directory for all log files
  log_level: "INFO"                                      # Log verbosity: DEBUG, INFO, WARNING, ERROR
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Python logging format
```

**Log Levels**:
- `DEBUG`: Detailed debugging information
- `INFO`: General information (recommended)
- `WARNING`: Warning messages only
- `ERROR`: Error messages only

### Phase-Specific Log Files

```yaml
logging:
  # Phase-specific log files (optional, logs to main if not specified)
  main_log: "pipeline_main.log"                          # Main pipeline execution log
  phase1_log: "phase1_schema_linking.log"                # Schema linking phase log
  phase2_log: "phase2_sql_generation.log"                # SQL generation phase log
  phase3_log: "phase3_sql_selection.log"                 # SQL selection phase log
  evaluation_log: "evaluation.log"                       # Evaluation results log
```

### Structured Logging

```yaml
logging:
  # Structured logging for analysis
  json_logs: true                                         # Enable JSON-formatted logs for analysis
  json_log_file: "pipeline_execution.json"               # Structured log file for machine parsing
  log_model_outputs: true                                 # Log complete model inputs/outputs for analysis
```

**Model Output Logging**:
- `log_model_outputs: true`: Captures all model prompts and responses
- Creates larger log files but enables detailed analysis
- Includes successful and failed model calls

### Prompt Inspection

```yaml
logging:
  # Prompt inspection and debugging
  save_prompts: true                                      # Save all prompts sent to models for inspection
  prompts_directory: "logs/prompts"                      # Directory for saved prompts (organized by phase)
```

**Prompt Files**: Organized by phase and model:
- `phase1_schema_refinement_20250101_120000_q1.txt`
- `phase2_sql_generation_xiyan_mschema_temp0.1_candidate1_20250101_120000_q1.txt`
- `phase3_sql_ranking_20250101_120000_q1.txt`

## Models Configuration

### Local Model Storage

```yaml
models:
  local_models_path: "data/models"                       # Cache directory for downloaded models
  enable_model_reuse: true                               # Reuse same models across different purposes to save memory
```

**Model Reuse**:
- `true`: Same model instance used for multiple purposes (saves memory)
- `false`: Separate model instances for each purpose

### Embedding Model

```yaml
models:
  embedding:
    model_name: "Qwen/Qwen3-Embedding-8B"                # HuggingFace model identifier
    device: "cuda"                                        # Device: "cuda", "cpu", or "auto"
    batch_size: 32                                        # Batch size for embedding computation
    max_length: 512                                       # Maximum token length for embeddings
```

**Device Options**:
- `"cuda"`: Use GPU (recommended)
- `"cpu"`: Use CPU only
- `"auto"`: Automatically select best available

### SQL Generation Models

```yaml
models:
  sql_generation:
    models:
      - name: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"  # Main SQL generation model
        type: "open_source"                               # Model type (always open_source for Phase 2)
        device: "cuda"                                    # GPU device for inference
        local_path: null                                  # Optional: Local directory path if model already downloaded
        max_new_tokens: 512                              # Maximum tokens to generate
        temperature_options: [0.1, 0.3, 0.7]            # Temperature values for diversity sampling
        
      - name: "Snowflake/Arctic-Text2SQL-R1-7B"          # Alternative SQL generation model
        type: "open_source"                               # Model type
        device: "cuda"                                    # GPU device for inference
        local_path: null                                  # Optional: Local directory path
        max_new_tokens: 512                              # Maximum tokens to generate
        temperature_options: [0.1, 0.3, 0.7]            # Temperature values
```

**Pre-downloaded Models Example**:
```yaml
models:
  sql_generation:
    models:
      - name: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
        type: "open_source"
        device: "cuda"
        local_path: "/home/user/models/xiyan-qwen-32b"  # Path to downloaded model directory
        max_new_tokens: 512
        temperature_options: [0.1, 0.3, 0.7]
```

### Commercial API Models

```yaml
models:
  commercial:
    gpt4o:
      api_key_env: "OPENAI_API_KEY"                      # Environment variable for API key
      api_key: "your-open-ai-api-key-here"               # Direct API key (use env var in production)
      model: "gpt-4o"                                    # OpenAI model identifier
      max_tokens: 1000                                   # Maximum response tokens
      temperature: 0.1                                   # Low temperature for consistent responses
      
    gemini:
      api_key_env: "GEMINI_API_KEY"                      # Environment variable for Gemini API key
      api_key: "your-gemini-api-key-here"                # Direct API key
      model: "gemini-1.5-pro"                           # Google Gemini model identifier
      max_tokens: 1000                                   # Maximum response tokens
      temperature: 0.1                                  # Low temperature for consistent responses
```

**API Key Priority**:
1. Environment variable (recommended for security)
2. Direct configuration (for testing only)

## Phase 1: Schema Linking Configuration

### HyTCSL Parameters

```yaml
phase1_schema_linking:
  # Hybrid Table-Column Schema Linking (HyTCSL) parameters
  top_k_tables: 10                                      # Maximum tables to consider initially
  top_k_columns_per_table: 5                           # Maximum columns per table to consider
```

**Tuning Guidelines**:
- **Small databases** (<20 tables): `top_k_tables: 5-8`
- **Medium databases** (20-50 tables): `top_k_tables: 8-12`
- **Large databases** (>50 tables): `top_k_tables: 12-15`

### Schema Expansion

```yaml
phase1_schema_linking:
  # Schema expansion for JOIN operations
  include_join_tables: true                             # Include tables needed for JOINs
  max_join_depth: 2                                     # Maximum depth for JOIN table expansion
```

**Join Depth Examples**:
- `1`: Direct joins only (A ↔ B)
- `2`: Two-hop joins (A ↔ B ↔ C)
- `3`: Three-hop joins (A ↔ B ↔ C ↔ D)

### Content/Value Linking

```yaml
phase1_schema_linking:
  # Content and value-based linking
  enable_value_linking: true                            # Match question values with database content
  value_similarity_threshold: 0.8                      # Similarity threshold for value matching (0-1)
```

**Value Linking**:
- Searches database for question-referenced values
- Higher threshold = more strict matching
- Can be computationally expensive for large databases

### LLM-Based Schema Refinement

```yaml
phase1_schema_linking:
  # LLM-based schema refinement (configurable model type)
  enable_llm_refinement: true                           # Use LLM to refine schema selection
  refinement_model: "gpt4o"                            # Model for refinement
  refinement_model_type: "commercial"                   # Model type: "commercial" or "local"
  refinement_model_path: null                          # Optional: Local directory path
  max_refinement_tokens: 2000                          # Maximum tokens for refinement prompt/response
  use_json_output: true                                # Use JSON output format for consistent parsing
```

**Model Options**:

**Commercial Models** (API cost but better performance):
```yaml
refinement_model: "gpt4o"                              # Use GPT-4o
refinement_model_type: "commercial"
# or
refinement_model: "gemini"                             # Use Gemini
refinement_model_type: "commercial"
```

**Local Models** (no API cost, uses GPU):
```yaml
refinement_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
refinement_model_type: "local"
refinement_model_path: null                            # Auto-download
# or
refinement_model_path: "/path/to/local/model"          # Pre-downloaded
```

### Database Descriptions

```yaml
phase1_schema_linking:
  # Database descriptions integration
  include_database_descriptions: true                   # Include detailed CSV descriptions in refinement prompts
  max_description_length: 8000                         # Maximum description length to avoid token overflow
```

**Description Sources**:
- CSV files in `database_description/` directories
- Natural language explanations of tables and columns
- Business context and data meanings

### Schema Representations

```yaml
phase1_schema_linking:
  # Schema representation formats for Phase 2
  schema_representations:
    - "m_schema"                                        # M-Schema format (academic standard)
    - "ddl"                                            # Data Definition Language (CREATE TABLE statements)
    - "json"                                           # JSON schema representation
    - "markdown"                                       # Human-readable markdown format
```

**Format Details**:
- **M-Schema**: Structured format with examples and metadata
- **DDL**: Standard SQL CREATE TABLE statements
- **JSON**: Machine-readable structured format
- **Markdown**: Human-readable table format

### Database Content Sampling

```yaml
phase1_schema_linking:
  # Database content sampling for context
  include_examples: true                                # Include sample data in schema representations
  max_examples_per_column: 3                          # Maximum example values per column
  include_statistics: true                             # Include basic statistics (count, distinct values)
```

## Phase 2: SQL Generation Configuration

### Candidate Generation Strategy

```yaml
phase2_sql_generation:
  # Candidate generation strategy
  candidates_per_model: 1                              # SQL candidates per model configuration
  enable_checkpointing: true                           # Save progress for long-running jobs
  checkpoint_interval: 5                             # Save checkpoint every N questions
```

**Checkpointing**:
- Saves generated candidates to resume interrupted sessions
- Especially important for long-running experiments
- Checkpoint files stored in `data/checkpoints/`

### Schema Representation Variants

```yaml
phase2_sql_generation:
  # Schema representation variants (must match Phase 1 representations)
  schema_representations:
    - "m_schema"                                        # M-Schema format
    - "ddl"                                            # DDL format
    - "json"                                           # JSON format
    - "markdown"                                       # Markdown format
```

**Important**: Must match Phase 1 `schema_representations` setting.

### Stochastic Sampling

```yaml
phase2_sql_generation:
  # Stochastic sampling for diversity
  temperature_values: [0.1, 0.3, 0.7]                 # Temperature values: low (conservative) to high (creative)
```

**Temperature Guide**:
- **0.1**: Very conservative, focuses on most likely tokens
- **0.3**: Balanced approach, some creativity
- **0.7**: More creative, diverse outputs
- **0.9**: Highly creative, potentially inconsistent

### Few-Shot Learning

```yaml
phase2_sql_generation:
  # Few-shot learning configuration
  enable_few_shot: true                                # Include example question-SQL pairs in prompts
  few_shot_examples: 3                                 # Number of examples to include
```

### SQL Validation and Fixing

```yaml
phase2_sql_generation:
  # SQL validation and automatic fixing
  enable_sql_validation: true                          # Validate generated SQL syntax
  enable_sql_fixing: true                              # Attempt to fix invalid SQL using LLM
  sql_fix_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"  # Model for SQL fixing
  sql_fix_model_type: "local"                          # Model type: "commercial" or "local"
  sql_fix_model_path: null                             # Optional: Local directory path
  max_fix_attempts: 3                                  # Maximum attempts to fix invalid SQL
```

**SQL Fixing Models**:

**Commercial Options** (cost ~$0.001-0.005 per fix):
```yaml
sql_fix_model: "gpt4o"                                 # Use GPT-4o for fixing
sql_fix_model_type: "commercial"
```

**Local Options** (no API cost, uses GPU):
```yaml
sql_fix_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
sql_fix_model_type: "local"
```

### Additional Sampling Parameters

```yaml
phase2_sql_generation:
  # Additional sampling parameters
  top_p: 0.9                                          # Nucleus sampling parameter (0-1)
  top_k: 50                                           # Top-k sampling parameter
```

## Phase 3: SQL Selection Configuration

### Candidate Filtering

```yaml
phase3_sql_selection:
  # Candidate filtering and deduplication
  remove_syntax_errors: true                           # Filter out syntactically invalid SQL
  remove_duplicates: true                              # Remove semantically duplicate candidates
  canonicalization_method: "sqlglot"                   # Method for SQL canonicalization: "sqlglot" or "simple"
```

**Canonicalization Methods**:
- `"sqlglot"`: Advanced parsing-based normalization (recommended)
- `"simple"`: Basic text normalization (fallback)

### LLM-Based Candidate Ranking

```yaml
phase3_sql_selection:
  # LLM-based candidate ranking (configurable model type)
  ranking_model: "gpt4o"                              # Model for ranking
  ranking_model_type: "commercial"                     # Model type: "commercial" or "local"
  ranking_model_path: null                            # Optional: Local directory path
  enable_chain_of_thought: true                        # Use chain-of-thought reasoning in ranking
  max_ranking_tokens: 1500                            # Maximum tokens for ranking prompt/response
  use_json_output: true                                # Use JSON output format for consistent parsing
```

**Ranking Model Options**:

**Commercial Models**:
```yaml
ranking_model: "gpt4o"                                 # Use GPT-4o for ranking
ranking_model_type: "commercial"
# or
ranking_model: "gemini"                                # Use Gemini for ranking
ranking_model_type: "commercial"
```

**Local Models**:
```yaml
ranking_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
ranking_model_type: "local"
```

### Voting and Consensus Mechanisms

```yaml
phase3_sql_selection:
  # Voting and consensus mechanisms
  enable_majority_voting: true                         # Consider popularity of similar candidates
  enable_execution_voting: true                        # Consider execution-based criteria (if available)
```

### Value Alignment Verification

```yaml
phase3_sql_selection:
  # Value alignment verification
  enable_value_alignment: true                         # Check if SQL references values from question
  value_alignment_threshold: 0.9                      # Threshold for value alignment score (0-1)
```

### Self-Consistency Checks

```yaml
phase3_sql_selection:
  # Self-consistency checks
  enable_self_consistency: true                        # Check consistency across different model outputs
  consistency_threshold: 0.7                          # Threshold for consistency score (0-1)
```

## Evaluation Configuration

### BIRD Benchmark Evaluation

```yaml
evaluation:
  # BIRD benchmark evaluation settings
  enable_execution_evaluation: true                    # Execute SQL against databases for accuracy
  execution_timeout: 30.0                             # Timeout for SQL execution (seconds)
  num_cpus: 1                                         # Number of CPU cores for parallel evaluation
```

**Parallel Evaluation**:
- `num_cpus: 1`: Sequential evaluation
- `num_cpus: 4`: Use 4 CPU cores for parallel evaluation
- `num_cpus: -1`: Use all available CPU cores

### Accuracy Analysis

```yaml
evaluation:
  # Detailed accuracy analysis
  calculate_by_difficulty: true                        # Break down accuracy by question difficulty
  calculate_by_database: true                          # Break down accuracy by database
```

### Usage Tracking

```yaml
evaluation:
  # Cost and usage tracking
  track_token_usage: true                             # Track API token consumption
  cost_tracking: true                                 # Calculate estimated API costs
```

### Output Formats

```yaml
evaluation:
  # Output formats
  save_detailed_results: true                         # Save complete pipeline results
  save_statistics: true                               # Save execution statistics
```

### Ground Truth References

```yaml
evaluation:
  # Ground truth references (for accuracy calculation)
  ground_truth_sql_path: "data/bird_benchmark/dev_gold.sql"  # Reference SQL queries
  difficulty_json_path: "data/bird_benchmark/dev.json"       # Question difficulty metadata
```

## Pipeline Execution Configuration

### Pipeline Control

```yaml
execution:
  # Pipeline control and subset processing
  max_questions: null                                   # Limit processing: null (all questions) or number
  start_from_checkpoint: false                         # Resume from latest checkpoint
  checkpoint_name: null                                # Specific checkpoint file to resume from
```

**Question Limiting**:
- `null`: Process all questions
- `100`: Process first 100 questions
- Combined with checkpointing for batch processing

### Resource Management

```yaml
execution:
  # Resource management and limits
  max_memory_gb: 32                                    # Maximum RAM usage (GB) - monitoring only
  max_gpu_memory_gb: 24                               # Maximum GPU memory (GB) - monitoring only
```

**Note**: These are monitoring limits, not hard constraints.

### Parallel Processing (Experimental)

```yaml
execution:
  # Parallel processing (experimental)
  enable_parallel_processing: false                    # Enable parallel question processing
  max_workers: 4                                       # Number of parallel workers
```

**Warning**: Parallel processing is experimental and may cause resource conflicts.

### Demo and Testing Modes

```yaml
execution:
  # Demo and testing modes
  demo_mode: false                                     # Enable single-question demo mode
  demo_question_id: 0                                 # Question ID for demo mode
```

### Progressive Metrics Display

```yaml
execution:
  # Progressive metrics display
  show_progressive_metrics: true                       # Show accuracy/execution metrics after each question
  progressive_metrics_frequency: 1                    # Show metrics every N questions (1=every question)
```

**Frequency Options**:
- `1`: Show metrics after every question (verbose)
- `5`: Show metrics every 5 questions
- `10`: Show metrics every 10 questions

### SQL Execution Results Logging

```yaml
execution:
  # SQL execution results logging
  show_sql_execution_results: true                    # Show SQL execution results in terminal
  max_result_rows_display: 3                          # Maximum rows to display in terminal (0=disable)
  log_sql_execution_to_json: true                     # Log detailed SQL execution results to JSON files
```

**Result Display**:
- Shows predicted vs ground truth execution results
- Useful for debugging incorrect answers
- Can be verbose for large result sets

### Model Outputs in Results

```yaml
execution:
  # Model outputs in results
  include_model_outputs_in_results: true              # Include all model raw outputs in pipeline_results.json
```

**Model Output Inclusion**:
- `true`: Includes all model prompts and responses in final JSON
- `false`: Excludes model outputs (smaller files)
- Useful for detailed analysis and debugging

### Session Management

```yaml
execution:
  # Session management (for cluster environments)
  max_session_time_hours: 3.5                         # Maximum session duration (for H100 clusters)
  auto_checkpoint_on_timeout: true                     # Automatically save checkpoint on timeout
```

## Advanced Configuration

### Caching and Performance

```yaml
advanced:
  # Caching and performance optimization
  enable_caching: true                                 # Cache intermediate results for faster reruns
  cache_directory: "data/cache"                       # Directory for cached data
```

### Error Handling

```yaml
advanced:
  # Error handling and robustness
  continue_on_error: true                              # Continue processing despite individual failures
  max_errors_per_phase: 10                            # Maximum errors before aborting phase
```

### System Monitoring

```yaml
advanced:
  # System monitoring and progress tracking
  enable_progress_bars: true                           # Show progress bars during execution
  enable_memory_monitoring: false                       # Monitor and log memory usage
  enable_gpu_monitoring: false                          # Monitor and log GPU usage
```

**Monitoring Features**:
- Progress bars for visual feedback
- Memory usage tracking (can impact performance)
- GPU utilization monitoring

### Reproducibility

```yaml
advanced:
  # Reproducibility and debugging
  random_seed: 42                                      # Random seed for reproducible results
  deterministic_mode: false                            # Enable deterministic model outputs (slower)
```

**Deterministic Mode**:
- `true`: Completely reproducible results (slower)
- `false`: Faster execution, some randomness

## Configuration Validation

### Required Fields

The following fields are required and will cause validation errors if missing:

```yaml
data:
  bird_benchmark_path: "..."    # Required
  dev_json_path: "..."         # Required
  databases_path: "..."        # Required

models:
  embedding:
    model_name: "..."          # Required
```

### Common Validation Errors

1. **Missing API Keys**: Commercial models require valid API keys
2. **Invalid Paths**: Data paths must point to valid directories
3. **Model Conflicts**: Schema representations must match between phases
4. **Resource Limits**: GPU memory settings must be realistic

### Configuration Testing

Test your configuration:

```bash
python -c "
from src.core.config_manager import ConfigManager
config = ConfigManager('configs/your_config.yaml')
print('Configuration valid!')
print(f'Using {len(config.get_model_config(\"sql_generation\"))} SQL models')
"
```

## Configuration Examples

### Minimal Cost Configuration

```yaml
# Minimal cost setup (local models only)
phase1_schema_linking:
  enable_llm_refinement: false

phase2_sql_generation:
  enable_sql_fixing: false
  temperature_values: [0.1]

phase3_sql_selection:
  ranking_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
  ranking_model_type: "local"
```

### High-Performance Configuration

```yaml
# High-performance setup (commercial models)
phase1_schema_linking:
  refinement_model: "gpt4o"
  refinement_model_type: "commercial"

phase2_sql_generation:
  sql_fix_model: "gpt4o"
  sql_fix_model_type: "commercial"

phase3_sql_selection:
  ranking_model: "gpt4o"
  ranking_model_type: "commercial"
```

### Development/Testing Configuration

```yaml
# Development configuration
execution:
  max_questions: 10
  demo_mode: false
  show_progressive_metrics: true

logging:
  log_level: "DEBUG"
  save_prompts: true
  log_model_outputs: true

advanced:
  enable_memory_monitoring: true
  enable_gpu_monitoring: true
```

## Best Practices

### 1. Start Small
Begin with limited questions and local models for testing:

```yaml
execution:
  max_questions: 5
models:
  # Use local models initially
```

### 2. Monitor Resources
Enable monitoring for resource-constrained environments:

```yaml
advanced:
  enable_memory_monitoring: true
  enable_gpu_monitoring: true
```

### 3. Use Checkpointing
For long experiments, enable frequent checkpointing:

```yaml
phase2_sql_generation:
  enable_checkpointing: true
  checkpoint_interval: 5
```

### 4. Track Costs
Enable cost tracking when using commercial models:

```yaml
evaluation:
  track_token_usage: true
  cost_tracking: true
```

### 5. Save Detailed Logs
For research analysis, save comprehensive logs:

```yaml
logging:
  save_prompts: true
  log_model_outputs: true
execution:
  include_model_outputs_in_results: true
```

For more configuration examples and troubleshooting, see:
- [Setup Guide](docs/SETUP_GUIDE.md)
- [Experiment Guide](docs/EXPERIMENT_GUIDE.md)