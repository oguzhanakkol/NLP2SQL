# Methodology Guide

This document provides a comprehensive technical overview of the Natural Language to SQL Pipeline methodology, including the theoretical foundation and implementation details of each phase.

## Overview

The pipeline implements a **three-phase approach** to natural language to SQL conversion:

1. **Phase 1**: Schema Linking & Pruning (HyTCSL + LLM Refinement)
2. **Phase 2**: SQL Candidate Generation (Multi-Model Ensemble)
3. **Phase 3**: SQL Selection (Multi-Criteria Ranking & Voting)

## Theoretical Foundation

### Problem Formulation

Given:
- Natural language question **Q**
- Database schema **S** = {T₁, T₂, ..., Tₙ} with tables and columns
- Database instance **D**

Goal: Generate SQL query **SQL** such that:
- **SQL** is syntactically valid
- **SQL(D)** produces the correct answer for **Q**
- **SQL** uses only relevant schema elements from **S**

### Key Challenges

1. **Schema Complexity**: Large databases with hundreds of tables
2. **Semantic Ambiguity**: Multiple valid interpretations of natural language
3. **Join Reasoning**: Complex relationships between tables
4. **Value Grounding**: Linking question values to database content
5. **SQL Syntax**: Generating syntactically correct and executable queries

## Phase 1: Schema Linking & Pruning

### 1.1 Hybrid Table-Column Schema Linking (HyTCSL)

Our approach combines multiple techniques for robust schema linking:

#### Embedding-Based Similarity

**Objective**: Find semantically similar schema elements to the question.

**Method**:
```
similarity(Q, element) = cosine(embed(Q), embed(element))
```

Where:
- `embed()` uses Qwen3-Embedding-8B model
- Elements include table names, column names, and descriptions

**Algorithm**:
1. Extract table information: `T = {(name, comment, searchable_text)}`
2. Extract column information: `C = {(table.column, type, comment, examples)}`
3. Compute embeddings for question and all schema elements
4. Rank by cosine similarity
5. Select top-k tables and top-k columns per table

#### Content/Value Linking

**Objective**: Ground question values in database content.

**Method**:
1. Extract potential values from question using regex patterns:
   - Quoted strings: `"value"` or `'value'`
   - Numbers: `\d+\.?\d*`
   - Capitalized words: `[A-Z][a-z]+`
2. Search database for matching values:
   ```sql
   SELECT COUNT(*) FROM table 
   WHERE LOWER(CAST(column AS TEXT)) LIKE '%value%'
   ```
3. Include tables/columns with matching values

#### Schema Expansion for Joins

**Objective**: Include necessary tables for join operations.

**Algorithm**:
1. Identify foreign key relationships from schema metadata
2. For each selected table, check connected tables via foreign keys
3. Include bridge tables needed for joins up to depth `max_join_depth`
4. Add primary key columns for join tables

### 1.2 LLM-Based Schema Refinement

**Objective**: Use large language models to refine schema selection with reasoning.

**Method**:
1. **Prompt Construction**:
   ```
   Question: {question}
   Evidence: {evidence}
   Database Schema: {mschema}
   Database Descriptions: {csv_descriptions}
   
   Task: Refine the initial schema selection...
   ```

2. **Model Selection**: Commercial (GPT-4o, Gemini) or Local (XiYanSQL)

3. **Output Format**: JSON for structured parsing
   ```json
   {
     "REFINED_TABLES": ["table1", "table2"],
     "REFINED_COLUMNS": ["table1.col1", "table2.col2"],
     "REASONING": "explanation of refinements"
   }
   ```

4. **Fallback**: Text parsing if JSON fails

### 1.3 Schema Representations

Generate multiple schema formats for Phase 2:

#### M-Schema Format
```
【DB_ID】 database_name
【Schema】
# Table: students, Information about students
[
(id:INTEGER, Primary Key),
(name:TEXT, Student full name),
(age:INTEGER, Student age, Examples: [18, 19, 20])
]
【Foreign keys】
enrollments.student_id=students.id
```

#### DDL Format
```sql
CREATE TABLE students (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER
);
```

#### JSON Format
```json
{
  "database": "university",
  "tables": {
    "students": {
      "columns": {
        "id": {"type": "INTEGER", "primary_key": true},
        "name": {"type": "TEXT", "examples": ["Alice", "Bob"]}
      }
    }
  }
}
```

## Phase 2: SQL Candidate Generation

### 2.1 Multi-Model Ensemble Approach

**Objective**: Generate diverse SQL candidates using multiple models and configurations.

**Models Supported**:
- **Open-Source**: XiYanSQL-QwenCoder-32B, Arctic-Text2SQL-7B
- **Commercial**: GPT-4o, Gemini (not used in Phase 2 by design)

**Configuration Space**:
- Models: M models
- Schema Representations: R representations  
- Temperatures: T temperature values
- **Total Configurations**: M × R × T

### 2.2 Stochastic Sampling Strategy

**Temperature Sampling**: Control creativity vs consistency
- **Low (0.1)**: Conservative, syntax-focused
- **Medium (0.3)**: Balanced approach
- **High (0.7)**: Creative, diverse solutions

**Sampling Parameters**:
```yaml
temperature_values: [0.1, 0.3, 0.7]
top_p: 0.9              # Nucleus sampling
candidates_per_model: 5  # Multiple samples per configuration
```

### 2.3 Few-Shot Prompting

**Prompt Structure**:
```
System: You are an expert SQL generator...

Example 1:
Question: How many students are there?
Schema: [M-Schema format]
SQL: SELECT COUNT(*) FROM students;

Example 2:
Question: List courses taught by CS professors.
Schema: [M-Schema format]  
SQL: SELECT c.name FROM courses c JOIN professors p ON...

Current Task:
Question: {question}
Schema: {schema_representation}
SQL:
```

### 2.4 SQL Validation and Fixing

**Validation Process**:
1. **Syntax Checking**: Use SQLGlot parser
2. **Basic Validation**: 
   - Non-empty query
   - Starts with SELECT/WITH
   - Valid structure

**Automatic Fixing**:
1. **Model Selection**: Commercial (GPT-4o) or Local (XiYan)
2. **Fix Prompt**:
   ```
   Fix the following SQL query:
   Original: {invalid_sql}
   Error: {error_message}
   Fixed SQL:
   ```
3. **Validation**: Re-validate fixed SQL
4. **Metadata**: Track which candidates were fixed

### 2.5 Candidate Pool Management

**Storage**: JSON serialization with metadata
```json
{
  "sql": "SELECT COUNT(*) FROM students",
  "model_name": "XiYanSQL",
  "schema_representation": "m_schema",
  "temperature": 0.1,
  "is_valid": true,
  "generation_time": 1.2,
  "was_fixed": false
}
```

**Checkpointing**: Resume generation from partial completion

## Phase 3: SQL Selection

### 3.1 Candidate Filtering

#### Validity Filtering
Remove syntactically invalid candidates based on `is_valid` flag.

#### Deduplication
**SQLGlot Canonicalization**:
```python
def canonicalize_sql(sql):
    parsed = parse_one(sql, dialect='sqlite')
    canonical = str(parsed).upper()
    return re.sub(r'\s+', ' ', canonical).strip()
```

**Grouping**: Candidates with identical canonical forms are grouped together.

### 3.2 Multi-Criteria Scoring

Each candidate receives scores across multiple dimensions:

#### Validity Score
```
validity_score = 1.0 if is_valid else 0.0
```

#### Popularity Score
```
popularity_score = count(similar_candidates) / total_candidates
```

#### LLM Critic Score

**Chain-of-Thought Ranking**:
```
Prompt: Analyze each SQL candidate considering:
1. Correctness: Does it answer the question?
2. Completeness: Are all necessary elements included?
3. Efficiency: Is it well-structured?
4. Schema adherence: Correct tables/columns?

Output: JSON with scores 0.0-1.0 for each candidate
```

**Commercial Models**: GPT-4o, Gemini with JSON output
**Local Models**: XiYan with text parsing

#### Value Alignment Score

**Algorithm**:
1. Extract values from question: quotes, numbers, entities
2. Check presence in SQL query (case-insensitive)
3. Calculate alignment ratio:
   ```
   alignment_score = matched_values / total_question_values
   ```

#### Self-Consistency Score

**Factors**:
- Model diversity: How many different models generated similar SQL
- Configuration diversity: Variety of generation configs
- Popularity weighting

**Formula**:
```
consistency_score = 0.4 × popularity + 0.3 × model_diversity + 0.3 × config_diversity
```

### 3.3 Final Selection

**Weighted Combination**:
```
final_score = w₁×validity + w₂×popularity + w₃×llm_critic + w₄×value_alignment + w₅×consistency
```

**Default Weights**:
- Validity: 0.2
- Popularity: 0.15  
- LLM Critic: 0.35
- Value Alignment: 0.15
- Self-Consistency: 0.15

**Selection**: Candidate with highest `final_score`

## Evaluation Methodology

### BIRD Benchmark Evaluation

**Execution Accuracy**:
```
accuracy = correct_executions / total_questions
```

Where `correct_execution` means:
1. Predicted SQL executes without error
2. Result matches ground truth result (set comparison)

**Metrics**:
- **Overall Accuracy**: Across all questions
- **Execution Success Rate**: Valid SQL percentage  
- **Difficulty Breakdown**: Simple/Moderate/Challenging
- **Database Breakdown**: Per-database performance

### Evaluation Process

1. **Load Ground Truth**: From BIRD dev.json
2. **Execute Predictions**: Against SQLite databases
3. **Execute Ground Truth**: For comparison
4. **Compare Results**: Set-based comparison handling row order
5. **Aggregate Metrics**: By difficulty, database, overall

## Implementation Details

### Embedding Model

**Model**: Qwen/Qwen3-Embedding-8B
- **Input**: Text (question, schema elements)
- **Output**: 768-dimensional embeddings
- **Similarity**: Cosine similarity for ranking

### SQL Generation Models

**XiYanSQL-QwenCoder-32B**:
- **Type**: Causal Language Model
- **Context**: 4K tokens
- **Generation**: Autoregressive with temperature sampling

**Arctic-Text2SQL-7B**:
- **Type**: Specialized Text-to-SQL model
- **Architecture**: Transformer-based
- **Training**: Optimized for SQL generation

### Commercial APIs

**GPT-4o**:
- **Provider**: OpenAI
- **Context**: 128K tokens
- **Features**: JSON mode, function calling
- **Cost**: ~$5/1M input tokens, $15/1M output tokens

**Gemini 1.5 Pro**:
- **Provider**: Google
- **Context**: 1M tokens
- **Features**: Multimodal capabilities
- **Cost**: Variable pricing

## Performance Optimizations

### Memory Management
- **Model Reuse**: Same model for multiple purposes
- **Batch Processing**: Process multiple items together
- **Cache Clearing**: Automatic cleanup to prevent OOM

### Checkpointing Strategy
- **Frequency**: Every N questions (configurable)
- **Content**: Progress state, generated candidates
- **Resume Logic**: Continue from last checkpoint

### Parallel Processing
- **Evaluation**: Multi-CPU result comparison
- **Generation**: Asynchronous model calls (where applicable)

## Theoretical Contributions

### 1. Hybrid Schema Linking (HyTCSL)
Combines embedding similarity, value grounding, and join reasoning for comprehensive schema selection.

### 2. Multi-Variant Ensemble
Systematic exploration of model×representation×temperature space for diverse candidate generation.

### 3. Multi-Criteria Selection
Principled combination of validity, popularity, expert judgment, value alignment, and consistency for robust selection.

### 4. Progressive Evaluation
Real-time accuracy tracking during pipeline execution for immediate feedback.

## Limitations and Future Work

### Current Limitations
1. **Scalability**: Large schemas may overwhelm context windows
2. **Complex Joins**: Multi-hop reasoning still challenging
3. **Ambiguity**: Multiple valid interpretations not fully addressed
4. **Cost**: Commercial model usage can be expensive

### Future Directions
1. **Retrieval-Augmented Generation**: For handling large schemas
2. **Program Synthesis**: More systematic SQL construction
3. **Interactive Clarification**: Handle ambiguous questions
4. **Fine-tuning**: Domain-specific model adaptation

## References

- BIRD Benchmark: [Paper citation]
- Schema Linking: [Related work citations]
- SQL Generation: [Model citations]
- Evaluation Methods: [Benchmark citations]

For implementation details, see:
- [Configuration Guide](docs/CONFIGURATION_GUIDE.md)
- [Experiment Guide](docs/EXPERIMENT_GUIDE.md)