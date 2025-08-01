# Experiment Guide

This guide covers how to design, run, and analyze experiments with the Natural Language to SQL Pipeline (NLP2SQL)

## Quick Start

### Running Your First Experiment

```bash
# Basic experiment with default settings
python main.py --max-questions 10

# Demo single question for debugging
python main.py --demo --demo-question-id 0

# Full pipeline with custom config
python main.py --config configs/experiment_config.yaml
```

## Experiment Types

### 1. Baseline Experiments

#### Local Models Only (Cost-Free)

```bash
# Create baseline configuration
cp configs/pipeline_config.yaml configs/baseline_local.yaml
```

Edit configuration:

```yaml
# Use only local models
phase1_schema_linking:
  refinement_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
  refinement_model_type: "local"

phase3_sql_selection:
  ranking_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
  ranking_model_type: "local"
```

Run experiment:

```bash
python main.py --config configs/baseline_local.yaml --max-questions 100
```

#### Commercial Models Baseline

```yaml
# Use commercial models for better performance
phase1_schema_linking:
  refinement_model: "gpt4o"
  refinement_model_type: "commercial"

phase3_sql_selection:
  ranking_model: "gpt4o"
  ranking_model_type: "commercial"
```

### 2. Ablation Studies

#### Disable Schema Refinement

```yaml
phase1_schema_linking:
  enable_llm_refinement: false
```

#### Disable SQL Fixing

```yaml
phase2_sql_generation:
  enable_sql_fixing: false
```

#### Single Model vs Ensemble

```yaml
models:
  sql_generation:
    models:
      - name: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
        # Remove other models for single-model experiment
```

#### Temperature Sensitivity

```yaml
phase2_sql_generation:
  temperature_values: [0.1]      # Conservative only
  # or
  temperature_values: [0.7]      # Creative only
  # or
  temperature_values: [0.1, 0.3, 0.5, 0.7, 0.9]  # Full range
```

### 3. Model Comparison Experiments

#### Compare SQL Generation Models

Create separate configs for each model:

**Config A** (`configs/xiyan_experiment.yaml`):
```yaml
models:
  sql_generation:
    models:
      - name: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
```

**Config B** (`configs/arctic_experiment.yaml`):
```yaml
models:
  sql_generation:
    models:
      - name: "Snowflake/Arctic-Text2SQL-R1-7B"
```

Run experiments:

```bash
python main.py --config configs/xiyan_experiment.yaml --max-questions 100
python main.py --config configs/arctic_experiment.yaml --max-questions 100
```

#### Compare Ranking Models

```yaml
# Experiment 1: GPT-4o ranking
phase3_sql_selection:
  ranking_model: "gpt4o"
  ranking_model_type: "commercial"

# Experiment 2: Gemini ranking
phase3_sql_selection:
  ranking_model: "gemini"
  ranking_model_type: "commercial"

# Experiment 3: Local model ranking
phase3_sql_selection:
  ranking_model: "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
  ranking_model_type: "local"
```

### 4. Schema Representation Experiments

Test different schema formats:

```yaml
phase1_schema_linking:
  schema_representations:
    - "m_schema"    # Only M-Schema
    # or
    - "ddl"        # Only DDL
    # or
    - "json"       # Only JSON
    # or
    - "markdown"   # Only Markdown
```

### 5. Difficulty-Based Analysis

#### Filter by Difficulty

Create configs for each difficulty level and run separately:

```python
# Create difficulty-specific datasets
from src.core.data_loader import BirdDataLoader
from src.core.config_manager import ConfigManager

config = ConfigManager()
loader = BirdDataLoader(config)
questions = loader.load_questions()

# Filter questions
simple_questions = [q for q in questions if q['difficulty'] == 'simple']
moderate_questions = [q for q in questions if q['difficulty'] == 'moderate']
challenging_questions = [q for q in questions if q['difficulty'] == 'challenging']

print(f"Simple: {len(simple_questions)}")
print(f"Moderate: {len(moderate_questions)}")
print(f"Challenging: {len(challenging_questions)}")
```

## Experimental Design

### 1. Controlled Variables

Always specify these for reproducibility:

```yaml
advanced:
  random_seed: 42
  deterministic_mode: true

execution:
  max_questions: 100          # Fixed dataset size
```

### 2. Independent Variables

Examples of what to vary:

- **Models**: Different SQL generation models
- **Temperatures**: Various sampling temperatures
- **Schema Representations**: Different format types
- **Selection Methods**: Different ranking approaches
- **Component Enabling**: Ablation study variables

### 3. Dependent Variables

Key metrics to measure:

- **Execution Accuracy**: Percentage of correct results
- **Execution Success Rate**: Percentage of valid SQL
- **Processing Time**: Time per question
- **API Costs**: Cost per question
- **Token Usage**: Input/output token counts

## Running Experiments

### 1. Sequential Experiments

Run multiple configurations:

```bash
#!/bin/bash
# Sequential experiment runner

configs=("baseline" "no_refinement" "single_model" "high_temp")

for config in "${configs[@]}"; do
    echo "Running experiment: $config"
    python main.py \
        --config "configs/${config}_config.yaml" \
        --max-questions 100
    
    # Move results
    mkdir -p "results/$config"
    mv data/results/* "results/$config/"
done
```

### 2. Parallel Experiments

For independent experiments:

```bash
# Run multiple experiments in parallel
python main.py --config configs/experiment_1.yaml --max-questions 50 &
python main.py --config configs/experiment_2.yaml --max-questions 50 &
python main.py --config configs/experiment_3.yaml --max-questions 50 &
wait
```

### 3. Cluster Experiments

#### Slurm Array Jobs

```bash
#!/bin/bash
#SBATCH --job-name=nlp2sql_experiments
#SBATCH --array=1-5
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

configs=("baseline" "no_refinement" "commercial" "local_only" "single_temp")
config=${configs[$SLURM_ARRAY_TASK_ID-1]}

python main.py --config "configs/${config}.yaml" --max-questions 200
```


## Result Analysis

### 1. Basic Analysis

After running experiments, analyze results:

```python
import json
import pandas as pd

# Load results
with open('data/results/pipeline_results_20250101_120000.json', 'r') as f:
    results = json.load(f)

# Extract key metrics
evaluation = results['evaluation']
print(f"Overall Accuracy: {evaluation['overall_accuracy']:.2f}%")
print(f"Execution Success Rate: {evaluation['execution_success_rate']:.2f}%")

# Difficulty breakdown
for difficulty, stats in evaluation['difficulty_breakdown'].items():
    print(f"{difficulty}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
```

### 2. Comparative Analysis

Compare multiple experiments:

```python
import matplotlib.pyplot as plt

# Load multiple experiment results
experiments = {
    'Baseline': 'results/baseline/pipeline_results.json',
    'No Refinement': 'results/no_refinement/pipeline_results.json',
    'Commercial Models': 'results/commercial/pipeline_results.json'
}

accuracies = {}
for name, path in experiments.items():
    with open(path, 'r') as f:
        data = json.load(f)
    accuracies[name] = data['evaluation']['overall_accuracy']

# Plot comparison
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values())
plt.title('Experiment Comparison - Overall Accuracy')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('experiment_comparison.png')
plt.show()
```

### 3. Statistical Analysis

```python
from scipy import stats
import numpy as np

# Compare two experiments statistically
def compare_experiments(results1, results2):
    # Extract accuracy for each question
    acc1 = [r['sql_execution']['comparison']['is_correct'] 
            for r in results1['pipeline_results']]
    acc2 = [r['sql_execution']['comparison']['is_correct'] 
            for r in results2['pipeline_results']]
    
    # Chi-square test
    correct1, correct2 = sum(acc1), sum(acc2)
    total1, total2 = len(acc1), len(acc2)
    
    contingency_table = [
        [correct1, total1 - correct1],
        [correct2, total2 - correct2]
    ]
    
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    print(f"Chi-square: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant: {p_value < 0.05}")

# Load and compare
with open('results/baseline/pipeline_results.json', 'r') as f:
    baseline = json.load(f)
with open('results/commercial/pipeline_results.json', 'r') as f:
    commercial = json.load(f)

compare_experiments(baseline, commercial)
```

## Experiment Templates

### 1. Model Performance Study

```yaml
# configs/model_study_template.yaml
execution:
  max_questions: 200
  show_progressive_metrics: true

logging:
  log_level: "INFO"
  save_prompts: true

advanced:
  random_seed: 42
  deterministic_mode: true

# Vary these parameters:
models:
  sql_generation:
    models:
      - name: "MODEL_NAME_HERE"
        temperature_options: [0.1, 0.3, 0.7]
```

### 2. Cost-Accuracy Trade-off Study

```yaml
# configs/cost_study_template.yaml
execution:
  max_questions: 100

evaluation:
  track_token_usage: true
  cost_tracking: true

# Compare configurations:
# 1. All local models (cost = $0)
# 2. Mixed local/commercial
# 3. All commercial models
```

### 3. Schema Representation Study

```yaml
# configs/schema_study_template.yaml
phase1_schema_linking:
  schema_representations:
    - "REPRESENTATION_TYPE_HERE"  # Vary this

phase2_sql_generation:
  schema_representations:
    - "REPRESENTATION_TYPE_HERE"  # Match phase 1
```

## Best Practices

### 1. Reproducibility

```yaml
# Always include these for reproducible experiments
advanced:
  random_seed: 42
  deterministic_mode: true

logging:
  save_prompts: true
  log_model_outputs: true

execution:
  include_model_outputs_in_results: true
```

### 2. Version Control

```bash
# Tag your experiments
git tag -a exp-baseline-v1.0 -m "Baseline experiment"
git tag -a exp-ablation-v1.0 -m "Ablation study"

# Create experiment branches
git checkout -b experiment/model-comparison
```

### 3. Documentation

Create experiment logs:

```python
# experiment_log.py
import json
from datetime import datetime

experiment_log = {
    "experiment_id": "model_comparison_001",
    "date": datetime.now().isoformat(),
    "description": "Compare XiYanSQL vs Arctic models",
    "hypothesis": "XiYanSQL should perform better on complex queries",
    "configuration": "configs/model_comparison.yaml",
    "results_path": "results/model_comparison/",
    "notes": "Initial experiment with 100 questions"
}

with open('experiment_log.json', 'w') as f:
    json.dump(experiment_log, f, indent=2)
```

### 4. Resource Management

Monitor resources during experiments:

```python
# Add to config for monitoring
advanced:
  enable_memory_monitoring: true
  enable_gpu_monitoring: true

logging:
  log_level: "DEBUG"  # For detailed resource logs
```

## Troubleshooting Experiments

### Common Issues

1. **Out of Memory**: Reduce batch sizes or model sizes
2. **API Rate Limits**: Add delays between requests
3. **Inconsistent Results**: Check random seed settings
4. **Long Runtime**: Use checkpointing and smaller datasets

### Debugging Failed Experiments

```bash
# Check logs
tail -f logs/pipeline_main.log

# Analyze checkpoints
python -c "
from src.core.checkpoint_manager import CheckpointManager
from src.core.config_manager import ConfigManager
config = ConfigManager()
checkpoint_manager = CheckpointManager(config)
print(checkpoint_manager.get_checkpoint_status())
"

# Resume from checkpoint
python main.py --resume --config configs/your_experiment.yaml
```

## Next Steps

1. **Design Your Experiment** - Define hypothesis and variables
2. **Create Configuration** - Set up experiment-specific config files
3. **Run Pilot Study** - Test with small dataset first
4. **Scale Up** - Run full experiments with proper resource allocation
5. **Analyze Results** - Use provided analysis scripts
6. **Document Findings** - Create reproducible research documentation

For more details, see:
- [Methodology Guide](docs/METHODOLOGY.md)
- [Configuration Guide](docs/CONFIGURATION_GUIDE.md)