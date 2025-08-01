import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import copy

import sqlglot
from sqlglot import parse_one, transpile


class SQLCandidate:
    
    def __init__(self, sql: str, model_name: str, schema_representation: str, 
                 temperature: float, **metadata):

        self.sql = sql.strip()
        self.model_name = model_name
        self.schema_representation = schema_representation
        self.temperature = temperature
        self.metadata = metadata
        
        self.is_valid = None
        self.validation_error = None
        self.execution_error = None
        
        self.generation_time = metadata.get('generation_time', 0.0)
        self.tokens_used = metadata.get('tokens_used', {})
        self.timestamp = metadata.get('timestamp', datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:

        return {
            'sql': self.sql,
            'model_name': self.model_name,
            'schema_representation': self.schema_representation,
            'temperature': self.temperature,
            'is_valid': self.is_valid,
            'validation_error': self.validation_error,
            'execution_error': self.execution_error,
            'generation_time': self.generation_time,
            'tokens_used': self.tokens_used,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SQLCandidate':
        candidate = cls(
            sql=data['sql'],
            model_name=data['model_name'],
            schema_representation=data['schema_representation'],
            temperature=data['temperature'],
            **data.get('metadata', {})
        )
        
        candidate.is_valid = data.get('is_valid')
        candidate.validation_error = data.get('validation_error')
        candidate.execution_error = data.get('execution_error')
        candidate.generation_time = data.get('generation_time', 0.0)
        candidate.tokens_used = data.get('tokens_used', {})
        candidate.timestamp = data.get('timestamp', datetime.now().isoformat())
        
        return candidate


class PromptBuilder:
    
    def __init__(self, config):

        self.config = config
        self.phase_config = config.get_phase_config(2)
        
        self.enable_few_shot = self.phase_config.get('enable_few_shot', True)
        self.few_shot_examples = self.phase_config.get('few_shot_examples', 3)
        
        self.example_pool = self._load_few_shot_examples()
    
    def build_prompt(self, question: str, schema_representation: str, 
                    schema_repr_type: str, evidence: str = "") -> str:

        prompt_parts = []
        
        prompt_parts.append(self._get_system_message(schema_repr_type))
        
        if self.enable_few_shot:
            examples = self._select_few_shot_examples(question, schema_repr_type)
            for example in examples:
                prompt_parts.append(self._format_example(example, schema_repr_type))
        
        prompt_parts.append(self._format_current_task(
            question, schema_representation, schema_repr_type, evidence
        ))
        
        return '\n\n'.join(prompt_parts)
    
    def _get_system_message(self, schema_repr_type: str) -> str:

        if schema_repr_type == 'm_schema':
            return """You are an expert SQL generator. Given a natural language question and a database schema in M-Schema format, generate a precise SQL query.

M-Schema format explanation:
- 【DB_ID】 indicates the database name
- 【Schema】 section contains table definitions
- Each table is formatted as: # Table: table_name, description
- Columns are listed in format: (column_name:TYPE, description, Primary Key if applicable, Examples: [sample values])
- 【Foreign keys】 section shows relationships as source.column=target.column

Guidelines:
1. Generate syntactically correct SQL
2. Use exact table and column names from the schema
3. Consider the data types and constraints
4. Use appropriate JOINs when accessing multiple tables
5. Include proper WHERE clauses for filtering
6. Use aggregate functions when needed
7. Pay attention to the question's specific requirements"""

        elif schema_repr_type == 'ddl':
            return """You are an expert SQL generator. Given a natural language question and database schema in DDL format, generate a precise SQL query.

Guidelines:
1. Generate syntactically correct SQL
2. Use exact table and column names from the CREATE TABLE statements
3. Consider the data types and constraints
4. Use appropriate JOINs when accessing multiple tables
5. Include proper WHERE clauses for filtering
6. Use aggregate functions when needed
7. Pay attention to the question's specific requirements"""

        elif schema_repr_type == 'json':
            return """You are an expert SQL generator. Given a natural language question and database schema in JSON format, generate a precise SQL query.

Guidelines:
1. Generate syntactically correct SQL
2. Use exact table and column names from the JSON schema
3. Consider the data types and constraints specified
4. Use appropriate JOINs when accessing multiple tables
5. Include proper WHERE clauses for filtering
6. Use aggregate functions when needed
7. Pay attention to the question's specific requirements"""

        else:
            return """You are an expert SQL generator. Given a natural language question and database schema in Markdown format, generate a precise SQL query.

Guidelines:
1. Generate syntactically correct SQL
2. Use exact table and column names from the Markdown tables
3. Consider the data types and constraints
4. Use appropriate JOINs when accessing multiple tables
5. Include proper WHERE clauses for filtering
6. Use aggregate functions when needed
7. Pay attention to the question's specific requirements"""
    
    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:

        examples = [
            {
                'question': "What is the total number of students in the database?",
                'schema_type': 'm_schema',
                'schema': """【DB_ID】 student_db
【Schema】
# Table: students
[
(student_id:INTEGER, Primary Key),
(name:TEXT, Student full name),
(age:INTEGER, Student age),
(grade:TEXT, Student grade level, Examples: [A, B, C, D, F])
]""",
                'sql': "SELECT COUNT(*) FROM students;",
                'explanation': "Simple count query on students table"
            },
            {
                'question': "List all courses taught by professors in the Computer Science department.",
                'schema_type': 'm_schema', 
                'schema': """【DB_ID】 university_db
【Schema】
# Table: professors
[
(prof_id:INTEGER, Primary Key),
(name:TEXT, Professor name),
(department:TEXT, Department name, Examples: [Computer Science, Mathematics, Physics])
]
# Table: courses
[
(course_id:INTEGER, Primary Key),
(course_name:TEXT, Course name),
(prof_id:INTEGER, Foreign key to professors table)
]
【Foreign keys】
courses.prof_id=professors.prof_id""",
                'sql': "SELECT c.course_name FROM courses c JOIN professors p ON c.prof_id = p.prof_id WHERE p.department = 'Computer Science';",
                'explanation': "JOIN query with filtering by department"
            }
        ]
        
        return examples
    
    def _select_few_shot_examples(self, question: str, schema_repr_type: str) -> List[Dict[str, Any]]:

        relevant_examples = [
            ex for ex in self.example_pool 
            if ex['schema_type'] == schema_repr_type
        ]
        
        return relevant_examples[:self.few_shot_examples]
    
    def _format_example(self, example: Dict[str, Any], schema_repr_type: str) -> str:
        return f"""Question: {example['question']}

Schema:
{example['schema']}

SQL: {example['sql']}"""
    
    def _format_current_task(self, question: str, schema_representation: str, 
                           schema_repr_type: str, evidence: str) -> str:

        task_parts = [f"Question: {question}"]
        
        if evidence:
            task_parts.append(f"Evidence/Hints: {evidence}")
        
        task_parts.extend([
            f"Schema:",
            schema_representation,
            "",
            "SQL:"
        ])
        
        return '\n'.join(task_parts)


class SQLValidator:
    
    def __init__(self, config):

        self.config = config
        self.phase_config = config.get_phase_config(2)
        
        self.enable_validation = self.phase_config.get('enable_sql_validation', True)
        self.enable_fixing = self.phase_config.get('enable_sql_fixing', True)
        self.max_fix_attempts = self.phase_config.get('max_fix_attempts', 3)
        
        if self.enable_fixing:
            fix_model_type = self.phase_config.get('sql_fix_model_type', 'commercial')
            if fix_model_type not in ['commercial', 'local']:
                raise ValueError(f"sql_fix_model_type must be 'commercial' or 'local', got: {fix_model_type}")
    
    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:

        if not self.enable_validation:
            return True, None
        
        try:
            parsed = parse_one(sql, dialect='sqlite')
            
            if not parsed:
                return False, "Failed to parse SQL"
            
            if not sql.strip():
                return False, "Empty SQL"
            
            if not sql.strip().upper().startswith(('SELECT', 'WITH')):
                return False, "SQL must start with SELECT or WITH"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def fix_sql(self, sql: str, model_manager, error_message: str, question_id: int = None, logger = None) -> Optional[str]:

        if not self.enable_fixing:
            return None
        
        try:
            fix_model = self.phase_config.get('sql_fix_model', 'XGenerationLab/XiYanSQL-QwenCoder-32B-2504')
            fix_model_type = self.phase_config.get('sql_fix_model_type', 'local')
            fix_model_path = self.phase_config.get('sql_fix_model_path', None)
            
            model = model_manager.load_sql_fixing_model(fix_model, fix_model_type, fix_model_path)
            if logger:
                logger.debug(f"Using {fix_model_type} model {fix_model} for SQL fixing")
            
            if fix_model_type == 'commercial':
                prompt = f"""You are an SQL expert. The following SQL query has a syntax error:

Original SQL:
{sql}

Error Message:
{error_message}

Please fix the SQL query and return only the corrected SQL without any explanation:"""
            else:
                prompt = f"""Fix the following SQL query that has a syntax error:

SQL: {sql}
Error: {error_message}

Fixed SQL:"""

            if logger:
                safe_model_name = fix_model.replace('/', '_')
                prompt_name = f'phase2_sql_fixing_{fix_model_type}_{safe_model_name}'
                logger.save_prompt(prompt_name, prompt, question_id)

            if fix_model_type == 'commercial':
                fixed_sql = model.generate(prompt, max_tokens=500)
            else:
                fixed_sql = model.generate(prompt, temperature=0.1, max_new_tokens=500)
            
            fixed_sql = self._clean_sql_response(fixed_sql)
            
            is_valid, _ = self.validate_sql(fixed_sql)
            
            if logger:
                logger.log_model_output(
                    question_id=question_id,
                    phase='phase2',
                    model_purpose='sql_fixing',
                    model_name=fix_model,
                    model_type=fix_model_type,
                    prompt=prompt,
                    raw_output=fixed_sql,
                    parsed_output={
                        'fixed_sql': fixed_sql,
                        'original_sql': sql,
                        'is_valid_after_fix': is_valid,
                        'fix_attempt': 1
                    },
                    output_format='text',
                    parsing_success=is_valid
                )
            
            if is_valid:
                return fixed_sql
            else:
                return None
                
        except Exception as e:
            if logger:
                error_message = str(e)
                logger.log_model_output(
                    question_id=question_id,
                    phase='phase2',
                    model_purpose='sql_fixing',
                    model_name=fix_model,
                    model_type=fix_model_type,
                    prompt=prompt if 'prompt' in locals() else "Error: Prompt not created",
                    raw_output=f"EXCEPTION: {error_message}",
                    parsed_output={
                        'success': False,
                        'error': error_message,
                        'original_sql': sql,
                        'exception_in_fixing_process': True
                    },
                    output_format='text',
                    parsing_success=False
                )
            return None
    
    def _clean_sql_response(self, response: str) -> str:

        response = response.strip()
        
        if response.startswith('```'):
            lines = response.split('\n')
            if len(lines) > 1:
                response = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
        
        lines = response.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Here', 'The', 'This', 'Note:', 'Explanation:')):
                sql_lines.append(line)
        
        return '\n'.join(sql_lines).strip()


class CandidatePoolManager:
    
    def __init__(self, config, checkpoint_manager):

        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.phase_config = config.get_phase_config(2)
        
        self.candidates_per_model = self.phase_config.get('candidates_per_model', 5)
        self.enable_checkpointing = self.phase_config.get('enable_checkpointing', True)
        self.checkpoint_interval = self.phase_config.get('checkpoint_interval', 100)
        
        if self.checkpoint_manager:
            self.checkpoint_data = self.checkpoint_manager.load_candidate_pools_checkpoint()
            if self.checkpoint_data:
                self.completed_configs = set(self.checkpoint_data.get('completed_configurations', []))
                self.candidate_pools = self.checkpoint_data.get('candidate_pools', {})
            else:
                self.completed_configs = set()
                self.candidate_pools = {}
        else:
            self.checkpoint_data = None
            self.completed_configs = set()
            self.candidate_pools = {}
    
    def generate_configuration_id(self, model_name: str, schema_repr: str, temperature: float) -> str:

        return f"{model_name}_{schema_repr}_{temperature:.1f}"
    
    def is_configuration_completed(self, config_id: str, question_id: int) -> bool:

        question_key = f"{question_id}_{config_id}"
        return question_key in self.completed_configs
    
    def save_candidate_pool(self, config_id: str, candidates: List[SQLCandidate], question_id: int) -> None:

        candidates_data = [candidate.to_dict() for candidate in candidates]
        
        if str(question_id) not in self.candidate_pools:
            self.candidate_pools[str(question_id)] = {}
        
        self.candidate_pools[str(question_id)][config_id] = candidates_data
        
        question_key = f"{question_id}_{config_id}"
        self.completed_configs.add(question_key)
        
        if self.enable_checkpointing and self.checkpoint_manager:
            checkpoint_data = {
                'completed_configurations': list(self.completed_configs),
                'candidate_pools': self.candidate_pools,
                'last_updated': datetime.now().isoformat()
            }
            self.checkpoint_manager.save_candidate_pools_checkpoint(checkpoint_data)
    
    def get_candidate_pool(self, question_id: int, config_id: str) -> Optional[List[SQLCandidate]]:

        if str(question_id) in self.candidate_pools:
            if config_id in self.candidate_pools[str(question_id)]:
                candidates_data = self.candidate_pools[str(question_id)][config_id]
                return [SQLCandidate.from_dict(data) for data in candidates_data]
        
        return None
    
    def get_all_candidates(self, question_id: int) -> List[SQLCandidate]:

        all_candidates = []
        
        if str(question_id) in self.candidate_pools:
            for config_id, candidates_data in self.candidate_pools[str(question_id)].items():
                candidates = [SQLCandidate.from_dict(data) for data in candidates_data]
                all_candidates.extend(candidates)
        
        return all_candidates
    
    def export_candidate_pools(self, output_path: str) -> None:
 
        export_data = {
            'completed_configurations': list(self.completed_configs),
            'candidate_pools': self.candidate_pools,
            'export_timestamp': datetime.now().isoformat(),
            'configuration_summary': {
                'total_configurations': len(self.completed_configs),
                'total_questions': len(self.candidate_pools),
                'total_candidates': sum(
                    len(configs) * self.candidates_per_model 
                    for configs in self.candidate_pools.values()
                )
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)


class SQLGenerator:
    
    def __init__(self, config, model_manager):

        self.config = config
        self.model_manager = model_manager
        self.phase_config = config.get_phase_config(2)
        self.logger = None
        
        self.prompt_builder = PromptBuilder(config)
        self.validator = SQLValidator(config)
        self.pool_manager = CandidatePoolManager(config, None)
        
        self.schema_representations = self.phase_config.get('schema_representations', ['m_schema'])
        self.temperature_values = self.phase_config.get('temperature_values', [0.1, 0.3, 0.7])
        self.candidates_per_model = self.phase_config.get('candidates_per_model', 5)
    
    def set_checkpoint_manager(self, checkpoint_manager):

        self.pool_manager = CandidatePoolManager(self.config, checkpoint_manager)
    
    def set_logger(self, logger):

        self.logger = logger
    
    def generate_candidates(self, question: Dict[str, Any], schema_result: Dict[str, Any]) -> Dict[str, Any]:

        question_id = question['question_id']
        question_text = question['question']
        evidence = question.get('evidence', '')
        
        sql_models = self.model_manager.get_available_sql_models()
        
        if not sql_models:
            if self.logger:
                self.logger.error("No SQL models available for candidate generation")
            raise ValueError("No SQL models available for candidate generation")
        
        total_configurations = len(sql_models) * len(self.schema_representations) * len(self.temperature_values)
        total_expected_candidates = total_configurations * self.candidates_per_model
        
        if self.logger:
            self.logger.log_candidate_generation_start(question_id, total_expected_candidates)
            self.logger.info(f"🤖 SQL Models for Phase 2: {', '.join(sql_models)}")
            self.logger.info(f"💡 Note: Commercial models (GPT/Gemini) are NOT used for SQL generation")
            
            print(f"🤖 SQL Models for Phase 2: {', '.join(sql_models)}")
            print(f"💡 Note: Commercial models (GPT/Gemini) are NOT used for SQL generation")
        
        all_candidates = []
        candidate_counter = 0
        generation_start_time = time.time()
        generation_summary = {
            'total_configurations': 0,
            'completed_configurations': 0,
            'failed_configurations': 0,
            'total_candidates': 0,
            'valid_candidates': 0,
            'models_used': []
        }
        
        for model_name in sql_models:
            if self.logger:
                self.logger.debug(f"Processing model: {model_name}")
            for schema_repr in self.schema_representations:
                for temperature in self.temperature_values:
                    
                    config_id = self.pool_manager.generate_configuration_id(
                        model_name, schema_repr, temperature
                    )
                    
                    generation_summary['total_configurations'] += 1
                    
                    if self.pool_manager.is_configuration_completed(config_id, question_id):
                        if self.logger:
                            self.logger.debug(f"Configuration {config_id} already completed, loading from pool")
                        candidates = self.pool_manager.get_candidate_pool(question_id, config_id)
                        if candidates:
                            all_candidates.extend(candidates)
                            generation_summary['completed_configurations'] += 1
                            generation_summary['total_candidates'] += len(candidates)
                            generation_summary['valid_candidates'] += sum(
                                1 for c in candidates if c.is_valid
                            )
                        continue
                    
                    try:
                        candidates = self._generate_for_configuration(
                            question_text, evidence, schema_result, 
                            model_name, schema_repr, temperature,
                            candidate_counter, question_id
                        )
                        
                        candidate_counter += len(candidates)
                        
                        all_candidates.extend(candidates)
                        generation_summary['completed_configurations'] += 1
                        generation_summary['total_candidates'] += len(candidates)
                        generation_summary['valid_candidates'] += sum(
                            1 for c in candidates if c.is_valid
                        )
                        
                        if model_name not in generation_summary['models_used']:
                            generation_summary['models_used'].append(model_name)
                        
                        self.pool_manager.save_candidate_pool(config_id, candidates, question_id)
                        
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to generate candidates for {config_id}: {str(e)}")
                        generation_summary['failed_configurations'] += 1
                        continue
        
        fixed_candidates = sum(1 for candidate in all_candidates if candidate.metadata.get('was_fixed', False))
        
        total_generation_time = time.time() - generation_start_time
        if self.logger:
            self.logger.log_candidate_generation_summary_enhanced(
                question_id,
                generation_summary['total_candidates'],
                generation_summary['valid_candidates'],
                generation_summary['total_candidates'] - generation_summary['valid_candidates'],
                fixed_candidates,
                total_generation_time,
                generation_summary['models_used']
            )
        
        result = {
            'question_id': question_id,
            'candidates': [candidate.to_dict() for candidate in all_candidates],
            'generation_summary': generation_summary,
            'configuration_details': {
                'schema_representations': self.schema_representations,
                'temperature_values': self.temperature_values,
                'candidates_per_model': self.candidates_per_model,
                'models_used': generation_summary['models_used']
            }
        }
        
        return result
    
    def _generate_for_configuration(self, question: str, evidence: str, schema_result: Dict[str, Any],
                                   model_name: str, schema_repr: str, temperature: float, 
                                   candidate_offset: int = 0, question_id: int = None) -> List[SQLCandidate]:
        candidates = []
        
        schema_representation = schema_result['schema_representations'].get(schema_repr, '')
        if not schema_representation:
            raise ValueError(f"Schema representation {schema_repr} not available")
        
        prompt = self.prompt_builder.build_prompt(
            question, schema_representation, schema_repr, evidence
        )
        
        try:
            model_config = self.config.get_model_config('sql_generation', model_name)
            local_path = model_config.get('local_path', None)
        except ValueError:
            local_path = None
        
        model = self.model_manager.load_sql_generation_model(model_name, local_path)
        
        for i in range(self.candidates_per_model):
            candidate_number = candidate_offset + i + 1
            
            try:
                start_time = time.time()
                
                if self.logger:
                    safe_model_name = model_name.replace('/', '_')
                    prompt_name = f'phase2_sql_generation_{safe_model_name}_{schema_repr}_temp{temperature:.1f}_candidate{i+1}'
                    self.logger.save_prompt(prompt_name, prompt, question_id)
                
                generated_sql = model.generate(
                    prompt, 
                    temperature=temperature,
                    max_new_tokens=512
                )
                
                generation_time = time.time() - start_time
                
                cleaned_sql = self._clean_generated_sql(generated_sql)
                
                if self.logger:
                    self.logger.log_model_output(
                        question_id=question_id,
                        phase='phase2',
                        model_purpose='sql_generation',
                        model_name=model_name,
                        model_type='local',
                        prompt=prompt,
                        raw_output=generated_sql,
                        parsed_output={
                            'cleaned_sql': cleaned_sql,
                            'schema_representation': schema_repr,
                            'temperature': temperature,
                            'generation_time': generation_time,
                            'candidate_index': i,
                            'success': True
                        },
                        output_format='text',
                        parsing_success=bool(cleaned_sql.strip())
                    )
                
                candidate = SQLCandidate(
                    sql=cleaned_sql,
                    model_name=model_name,
                    schema_representation=schema_repr,
                    temperature=temperature,
                    generation_time=generation_time,
                    candidate_index=i
                )
                
                is_valid, error_msg = self.validator.validate_sql(cleaned_sql)
                candidate.is_valid = is_valid
                candidate.validation_error = error_msg
                
                if self.logger:
                    self.logger.log_candidate_generated(
                        candidate_number,
                        model_name,
                        schema_repr,
                        temperature,
                        cleaned_sql,
                        is_valid,
                        error_msg,
                        generation_time
                    )
                
                fixed_flag = False
                if not is_valid and error_msg:
                    if self.logger:
                        print(f"\n{'='*45}")
                        print(f"🔧 SQL FIX ATTEMPT - CANDIDATE {candidate_number}")
                        print(f"{'='*45}")
                        print(f"Original SQL: {cleaned_sql[:80]}{'...' if len(cleaned_sql) > 80 else ''}")
                        print(f"Error: {error_msg}")
                        fix_model = self.phase_config.get('sql_fix_model', 'gpt4o')
                        print(f"Attempting fix using {fix_model}...")
                        print(f"{'-'*45}")
                        
                        self.logger.info(f"Starting SQL fix for candidate {candidate_number}", phase='phase2')
                    
                    fixed_sql = self.validator.fix_sql(cleaned_sql, self.model_manager, error_msg, question_id, self.logger)
                    if fixed_sql:
                        candidate.sql = fixed_sql
                        is_valid_after_fix, _ = self.validator.validate_sql(fixed_sql)
                        candidate.is_valid = is_valid_after_fix
                        candidate.metadata['was_fixed'] = True
                        candidate.metadata['original_sql'] = cleaned_sql
                        fixed_flag = True
                        
                        if self.logger:
                            status = "✅ FIXED & VALID" if is_valid_after_fix else "❌ STILL INVALID"
                            print(f"🔧 RESULT: {status}")
                            if is_valid_after_fix:
                                print(f"✅ Fixed SQL: {fixed_sql}")
                            else:
                                print(f"❌ Fixed SQL: {fixed_sql}")
                                print(f"❌ Fix unsuccessful - SQL still invalid")
                            print(f"{'='*45}\n")
                            
                            self.logger.info(f"SQL fix result: {status}", phase='phase2', extra_data={
                                'candidate_number': candidate_number,
                                'original_sql': cleaned_sql,
                                'fixed_sql': fixed_sql,
                                'fix_successful': is_valid_after_fix,
                                'sql_fix_applied': True
                            })
                    else:
                        if self.logger:
                            print(f"❌ RESULT: FIX FAILED")
                            print(f"❌ No valid fix could be generated")
                            print(f"{'='*45}\n")
                            self.logger.info(f"SQL fix failed for candidate {candidate_number}", phase='phase2')
                
                candidates.append(candidate)
                
            except Exception as e:
                if self.logger:
                    error_message = str(e)
                    self.logger.log_model_output(
                        question_id=question_id,
                        phase='phase2',
                        model_purpose='sql_generation',
                        model_name=model_name,
                        model_type='local',
                        prompt=prompt,
                        raw_output=f"ERROR: {error_message}",
                        parsed_output={
                            'success': False,
                            'error': error_message,
                            'schema_representation': schema_repr,
                            'temperature': temperature,
                            'candidate_index': i,
                            'generation_failed': True
                        },
                        output_format='text',
                        parsing_success=False
                    )
                
                candidate = SQLCandidate(
                    sql="",
                    model_name=model_name,
                    schema_representation=schema_repr,
                    temperature=temperature,
                    generation_error=str(e),
                    candidate_index=i
                )
                candidate.is_valid = False
                candidate.validation_error = f"Generation failed: {str(e)}"
                
                if self.logger:
                    self.logger.log_candidate_generated(
                        candidate_number,
                        model_name,
                        schema_repr,
                        temperature,
                        "",
                        False,
                        f"Generation failed: {str(e)}",
                        0.0
                    )
                
                candidates.append(candidate)
        
        return candidates
    
    def _clean_generated_sql(self, generated_text: str) -> str:

        cleaned = generated_text.strip()
        
        if not cleaned:
            return ""
        
        import re
        simple_sql_pattern = r'(SELECT\s+.*?)(?=\s*(?:```|\n\s*This\s+SQL|\n\s*The\s+above|\n\s*Note:|\n\s*Explanation:|\s*$))'
        simple_match = re.search(simple_sql_pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if simple_match:
            candidate_sql = simple_match.group(1).strip()
            if candidate_sql.endswith(';'):
                candidate_sql = candidate_sql[:-1].strip()
            if re.search(r'\b(FROM|WHERE|SELECT)\b', candidate_sql, re.IGNORECASE):
                return candidate_sql
        
        if '```sql' in cleaned.lower() or '```' in cleaned:
            code_block_pattern = r'```(?:sql)?\s*(.*?)```'
            matches = re.findall(code_block_pattern, cleaned, re.DOTALL | re.IGNORECASE)
            if matches:
                cleaned = matches[0].strip()
        # Need something more robust to extract SQL from explanations
        explanation_prefixes = [
            'Based on the provided schema and question, the following SQL query is generated:',
            'Based on the provided schema and question, the SQL query should',
            'Here is the SQL query for the given question:',
            'The SQL query for this question is:',
            'In the above SQL query, I have corrected',
            'In the above SQL query,',
            'In this SQL query,',
            'The above query',
            'This query',
            'SQL Query:',
            'Query:',
            'Answer:',
            'Result:',
            'Here\'s the SQL:',
            'The query is:',
            'SQL:',
            'Based on the provided schema',
            'According to the schema',
            'Given the schema'
        ]
        
        for prefix in explanation_prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        explanation_indicators = [
            'focus on calculating',
            'should focus on',
            'the following SQL',
            'this query will',
            'this will help',
            'note that',
            'please note'
        ]
        
        lines = cleaned.split('\n')
        sql_lines = []
        found_sql_start = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('--'):
                continue
            
            line_lower = line.lower()
            is_explanation = any(indicator in line_lower for indicator in explanation_indicators)
            
            if is_explanation and not any(sql_keyword in line_lower for sql_keyword in ['select', 'from', 'where', 'join', 'group', 'order', 'having', 'with']):
                continue
            # Same more robust check for explanations
            explanation_starters = [
                'the query', 'this query', 'here', 'explanation:', 'note:', 
                'this sql', 'the sql', 'this will', 'the result', 'we need to',
                'to answer', 'in this', 'for this', 'based on'
            ]
            
            if any(line_lower.startswith(starter) for starter in explanation_starters):
                if not any(sql_keyword in line_lower for sql_keyword in ['select', 'from', 'where']):
                    continue
            
            sql_indicators = ['select', 'with', 'from', 'where', 'join', 'group by', 'order by', 'having']
            if any(indicator in line_lower for indicator in sql_indicators):
                found_sql_start = True
            
            if found_sql_start or any(indicator in line_lower for indicator in sql_indicators):
                sql_lines.append(line)
        
        if not sql_lines:
            words = cleaned.split()
            sql_start_idx = -1
            
            for i, word in enumerate(words):
                if word.upper() in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']:
                    sql_start_idx = i
                    break
            
            if sql_start_idx != -1:
                remaining_words = words[sql_start_idx:]
                sql_words = []
                
                for word in remaining_words:
                    if any(explanation in word.lower() for explanation in [
                        'however', 'corrected', 'ensured', 'aligns', 'above'
                    ]):
                        if len(sql_words) > 3 and any(end_word in word.lower() for end_word in [
                            'however,', 'corrected', 'ensured', 'above,'
                        ]):
                            break
                    sql_words.append(word)
                
                result = ' '.join(sql_words)
            else:
                import re
                sql_pattern = r'(SELECT.*?(?:;|$|However|Corrected|Above))'
                match = re.search(sql_pattern, cleaned, re.IGNORECASE | re.DOTALL)
                if match:
                    result = match.group(1)
                    for end_word in ['However', 'Corrected', 'Above']:
                        if result.upper().endswith(end_word.upper()):
                            result = result[:-len(end_word)].strip()
                else:
                    result = cleaned
        else:
            result = '\n'.join(sql_lines).strip()
        
        if '. ' in result:
            parts = result.split('. ')
            for part in parts:
                if any(keyword in part.upper() for keyword in ['SELECT', 'FROM', 'WHERE']):
                    result = part
                    break
        
        if result.endswith(';'):
            result = result[:-1].strip()
        
        if ' -- ' in result:
            result = result.split(' -- ')[0].strip()
        
        return result
