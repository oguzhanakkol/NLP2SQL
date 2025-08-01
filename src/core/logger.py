import os
import json
import logging
import logging.handlers
import torch
import psutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
from collections import defaultdict

class PipelineLogger:
    
    def __init__(self, config_manager):

        self.config = config_manager
        self.log_config = config_manager.get_logging_config()
        
        self.log_dir = Path(self.log_config.get('log_directory', 'logs'))
        self.prompts_dir = Path(self.log_config.get('prompts_directory', 'logs/prompts'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_main_logger()
        self._setup_phase_loggers()
        self._setup_json_logger()
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_logs = []
        
        self.enable_gpu_monitoring = config_manager.get('advanced.enable_gpu_monitoring', True)
        self.enable_memory_monitoring = config_manager.get('advanced.enable_memory_monitoring', True)
        
        self.info("Pipeline logger initialized")
        self.log_system_info()
    
    def _setup_main_logger(self) -> None:

        self.main_logger = logging.getLogger('pipeline_main')
        self.main_logger.setLevel(getattr(logging, self.log_config.get('log_level', 'INFO')))
        
        self.main_logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.main_logger.addHandler(console_handler)
        
        main_log_file = self.log_dir / self.log_config.get('main_log', 'pipeline_main.log')
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_format = logging.Formatter(self.log_config.get('log_format'))
        file_handler.setFormatter(file_format)
        self.main_logger.addHandler(file_handler)
    
    def _setup_phase_loggers(self) -> None:

        self.phase_loggers = {}
        
        phase_logs = {
            'phase1': self.log_config.get('phase1_log', 'phase1_schema_linking.log'),
            'phase2': self.log_config.get('phase2_log', 'phase2_sql_generation.log'),
            'phase3': self.log_config.get('phase3_log', 'phase3_sql_selection.log'),
            'evaluation': self.log_config.get('evaluation_log', 'evaluation.log')
        }
        
        for phase_name, log_file in phase_logs.items():
            logger = logging.getLogger(f'pipeline_{phase_name}')
            logger.setLevel(getattr(logging, self.log_config.get('log_level', 'INFO')))
            logger.handlers.clear()
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / log_file, maxBytes=10*1024*1024, backupCount=3
            )
            file_format = logging.Formatter(self.log_config.get('log_format'))
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
            self.phase_loggers[phase_name] = logger
    
    def _setup_json_logger(self) -> None:

        if self.log_config.get('json_logs', True):
            self.json_log_file = self.log_dir / self.log_config.get('json_log_file', 'pipeline_execution.json')
            self.json_logs = []
    
    def info(self, message: str, phase: Optional[str] = None, extra_data: Optional[Dict] = None) -> None:
        self._log(logging.INFO, message, phase, extra_data)
    
    def debug(self, message: str, phase: Optional[str] = None, extra_data: Optional[Dict] = None) -> None:
        self._log(logging.DEBUG, message, phase, extra_data)
    
    def warning(self, message: str, phase: Optional[str] = None, extra_data: Optional[Dict] = None) -> None:
        self._log(logging.WARNING, message, phase, extra_data)
    
    def error(self, message: str, phase: Optional[str] = None, extra_data: Optional[Dict] = None) -> None:
        self._log(logging.ERROR, message, phase, extra_data)
    
    def _log(self, level: int, message: str, phase: Optional[str] = None, extra_data: Optional[Dict] = None) -> None:

        self.main_logger.log(level, message)
        
        if phase and phase in self.phase_loggers:
            self.phase_loggers[phase].log(level, message)
        
        if self.log_config.get('json_logs', True):
            json_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'level': logging.getLevelName(level),
                'message': message,
                'phase': phase,
                'extra_data': extra_data or {}
            }
            self.json_logs.append(json_entry)
    
    def log_question_start(self, question_id: int, question: str, db_id: str) -> None:
        self.info(f"Starting question {question_id}: {question[:100]}...", extra_data={
            'question_id': question_id,
            'db_id': db_id,
            'question_full': question
        })
    
    def log_phase_start(self, phase: str, question_id: int) -> None:
        self.info(f"Starting {phase} for question {question_id}", phase=phase, extra_data={
            'question_id': question_id,
            'phase_start': True
        })
    
    def log_phase_end(self, phase: str, question_id: int, duration: float, success: bool = True) -> None:
        status = "completed" if success else "failed"
        self.info(f"Phase {phase} {status} for question {question_id} in {duration:.2f}s", 
                 phase=phase, extra_data={
            'question_id': question_id,
            'phase_end': True,
            'duration': duration,
            'success': success
        })
    
    def log_model_usage(self, model_name: str, model_type: str, input_tokens: int, 
                       output_tokens: int, cost: Optional[float] = None) -> None:
        self.info(f"Model usage - {model_name}: {input_tokens} input, {output_tokens} output tokens", 
                 extra_data={
            'model_name': model_name,
            'model_type': model_type,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'usage_tracking': True
        })
    
    def log_commercial_api_call(self, api_name: str, endpoint: str, cost: Optional[float] = None) -> None:
        self.info(f"Commercial API call to {api_name} ({endpoint})", extra_data={
            'api_name': api_name,
            'endpoint': endpoint,
            'cost': cost,
            'commercial_api': True
        })
    
    def log_sql_generation(self, question_id: int, model_name: str, schema_repr: str, 
                          temperature: float, generated_sql: str, valid: bool) -> None:
        self.info(f"SQL generated by {model_name} for Q{question_id} (temp={temperature}, valid={valid})", 
                 phase='phase2', extra_data={
            'question_id': question_id,
            'model_name': model_name,
            'schema_representation': schema_repr,
            'temperature': temperature,
            'generated_sql': generated_sql,
            'valid_sql': valid,
            'sql_generation': True
        })
    
    def log_sql_selection(self, question_id: int, selected_sql: str, selection_method: str, 
                         candidates_count: int, selection_confidence: Optional[float] = None) -> None:
        self.info(f"SQL selected for Q{question_id} using {selection_method} from {candidates_count} candidates", 
                 phase='phase3', extra_data={
            'question_id': question_id,
            'selected_sql': selected_sql,
            'selection_method': selection_method,
            'candidates_count': candidates_count,
            'selection_confidence': selection_confidence,
            'sql_selection': True
        })
    
    def log_schema_linking(self, question_id: int, selected_tables: list, selected_columns: list, 
                          refinement_used: bool) -> None:
        self.info(f"Schema linked for Q{question_id}: {len(selected_tables)} tables, {len(selected_columns)} columns", 
                 phase='phase1', extra_data={
            'question_id': question_id,
            'selected_tables': selected_tables,
            'selected_columns': selected_columns,
            'refinement_used': refinement_used,
            'schema_linking': True
        })
    
    def log_checkpoint(self, checkpoint_name: str, question_id: int) -> None:
        self.info(f"Checkpoint saved: {checkpoint_name} at question {question_id}", extra_data={
            'checkpoint_name': checkpoint_name,
            'question_id': question_id,
            'checkpoint': True
        })
    
    def log_sql_execution_results(self, question_id: int, predicted_sql: str, 
                                 predicted_result: Optional[list], predicted_error: Optional[str],
                                 ground_truth_sql: Optional[str] = None, 
                                 ground_truth_result: Optional[list] = None,
                                 ground_truth_error: Optional[str] = None,
                                 is_correct: Optional[bool] = None,
                                 execution_success: bool = False) -> None:

        log_data = {
            'question_id': question_id,
            'predicted_sql': predicted_sql,
            'execution_success': execution_success,
            'predicted_result_count': len(predicted_result) if predicted_result else 0,
            'is_correct': is_correct,
            'sql_execution_results': True
        }
        
        if predicted_result is not None:
            log_data['predicted_execution_result'] = predicted_result
        if predicted_error:
            log_data['predicted_execution_error'] = predicted_error
            
        if ground_truth_sql:
            log_data['ground_truth_sql'] = ground_truth_sql
            log_data['ground_truth_result_count'] = len(ground_truth_result) if ground_truth_result else 0
            
        if ground_truth_result is not None:
            log_data['ground_truth_execution_result'] = ground_truth_result
        if ground_truth_error:
            log_data['ground_truth_execution_error'] = ground_truth_error
        
        status = "successful" if execution_success else "failed"
        accuracy = ""
        if is_correct is not None:
            accuracy = f", {'correct' if is_correct else 'incorrect'} result"
        
        message = f"SQL execution for Q{question_id}: {status}{accuracy}"
        
        self.info(message, phase='evaluation', extra_data=log_data)
    
    def save_prompt(self, prompt_name: str, prompt_content: str, question_id: Optional[int] = None) -> str:

        if not self.log_config.get('save_prompts', True):
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prompt_name}_{timestamp}"
        if question_id is not None:
            filename += f"_q{question_id}"
        filename += ".txt"
        
        prompt_path = self.prompts_dir / filename
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
        
        self.debug(f"Prompt saved: {prompt_path}")
        return str(prompt_path)
    
    def save_json_logs(self) -> None:
        if self.json_logs and self.log_config.get('json_logs', True):
            with open(self.json_log_file, 'w') as f:
                json.dump(self.json_logs, f, indent=2)
            self.info(f"JSON logs saved to {self.json_log_file}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'total_log_entries': len(self.json_logs),
            'log_directory': str(self.log_dir),
            'json_log_file': str(self.json_log_file) if hasattr(self, 'json_log_file') else None
        }
    
    @contextmanager
    def phase_context(self, phase: str, question_id: int):"
        start_time = datetime.now()
        self.log_phase_start(phase, question_id)
        
        try:
            yield
            duration = (datetime.now() - start_time).total_seconds()
            self.log_phase_end(phase, question_id, duration, success=True)
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.error(f"Phase {phase} failed for question {question_id}: {str(e)}", phase=phase)
            self.log_phase_end(phase, question_id, duration, success=False)
            raise
    
    def log_system_info(self) -> None:
        system_info = self.get_system_info()
        self.info("System Information:", extra_data=system_info)
        
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        print(f"CPU Count: {system_info['cpu_count']}")
        print(f"Memory Total: {system_info['memory_total_gb']:.1f} GB")
        print(f"Memory Available: {system_info['memory_available_gb']:.1f} GB")
        
        if system_info['gpu_available']:
            print(f"GPU Available: Yes")
            print(f"GPU Count: {system_info['gpu_count']}")
            for i, gpu_info in enumerate(system_info['gpu_details']):
                print(f"  GPU {i}: {gpu_info['name']} ({gpu_info['memory_total']:.1f} GB)")
        else:
            print("GPU Available: No")
        print("="*60 + "\n")
    
    def get_system_info(self) -> Dict[str, Any]:
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': 0,
            'gpu_details': []
        }
        
        if torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info['gpu_details'].append({
                    'device_id': i,
                    'name': props.name,
                    'memory_total': props.total_memory / (1024**3),
                    'memory_allocated': torch.cuda.memory_allocated(i) / (1024**3),
                    'memory_reserved': torch.cuda.memory_reserved(i) / (1024**3)
                })
        
        return info
    
    def log_gpu_usage(self, context: str = "") -> None:
        if not self.enable_gpu_monitoring or not torch.cuda.is_available():
            return
        
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024**3)
            
            utilization = (allocated / total) * 100 if total > 0 else 0
            
            gpu_info.append({
                'device_id': i,
                'name': props.name,
                'memory_allocated_gb': allocated,
                'memory_reserved_gb': reserved,
                'memory_total_gb': total,
                'utilization_percent': utilization
            })
        
        context_msg = f" ({context})" if context else ""
        self.info(f"GPU Usage{context_msg}", extra_data={
            'gpu_usage': True,
            'context': context,
            'gpu_details': gpu_info
        })
        
        print(f"\n🎮 GPU Usage{context_msg}:")
        for gpu in gpu_info:
            print(f"  GPU {gpu['device_id']} ({gpu['name']}): "
                  f"{gpu['memory_allocated_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB "
                  f"({gpu['utilization_percent']:.1f}%)")
    
    def log_memory_usage(self, context: str = "") -> None:
        if not self.enable_memory_monitoring:
            return
        
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
        
        context_msg = f" ({context})" if context else ""
        self.info(f"Memory Usage{context_msg}", extra_data={
            'memory_usage': True,
            'context': context,
            'memory_info': memory_info
        })
        
        print(f"💾 Memory Usage{context_msg}: "
              f"{memory_info['used_gb']:.1f}/{memory_info['total_gb']:.1f} GB "
              f"({memory_info['percent']:.1f}%)")
    
    def log_phase_start_detailed(self, phase: str, question_id: int, question: str = "") -> None:
        self.info(f"\n{'='*60}")
        self.info(f"STARTING {phase.upper()}")
        self.info(f"Question ID: {question_id}")
        if question:
            self.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        self.info(f"{'='*60}")
        
        self.log_gpu_usage(f"Phase Start: {phase}")
        self.log_memory_usage(f"Phase Start: {phase}")
        
        self.log_phase_start(phase, question_id)
    
    def log_phase_end_detailed(self, phase: str, question_id: int, duration: float, 
                             success: bool = True, result_summary: str = "") -> None:
        status = "✅ COMPLETED" if success else "❌ FAILED"
        self.info(f"\n{phase.upper()} {status}")
        self.info(f"Duration: {duration:.2f}s")
        if result_summary:
            self.info(f"Result: {result_summary}")
        
        self.log_gpu_usage(f"Phase End: {phase}")
        self.log_memory_usage(f"Phase End: {phase}")
        self.info(f"{'='*60}\n")
        
        self.log_phase_end(phase, question_id, duration, success)
    
    def log_model_loading(self, model_name: str, model_type: str) -> None:
        self.info(f"🔄 Loading {model_type} model: {model_name}")
        self.log_gpu_usage(f"Before loading {model_name}")
    
    def log_model_loaded(self, model_name: str, model_type: str) -> None:
        self.info(f"✅ Loaded {model_type} model: {model_name}")
        self.log_gpu_usage(f"After loading {model_name}")
    
    def log_progress(self, current: int, total: int, description: str = "") -> None:
        percentage = (current / total) * 100 if total > 0 else 0
        filled = int(50 * current // total) if total > 0 else 0
        bar = "█" * filled + "-" * (50 - filled)
        
        desc_msg = f" {description}" if description else ""
        self.info(f"Progress{desc_msg}: |{bar}| {current}/{total} ({percentage:.1f}%)")
        
        print(f"\r📊 Progress{desc_msg}: |{bar}| {current}/{total} ({percentage:.1f}%)", end="", flush=True)
        
        if current == total:
            print()
    
    def log_candidate_generation_start(self, question_id: int, total_candidates: int) -> None:
        self.info(f"\n🔄 GENERATING SQL CANDIDATES")
        self.info(f"Question ID: {question_id}")
        self.info(f"Expected candidates: {total_candidates}")
        self.info(f"{'='*60}")
        
        print(f"\n🔄 GENERATING SQL CANDIDATES")
        print(f"Question ID: {question_id}")
        print(f"Expected candidates: {total_candidates}")
        print(f"{'='*60}")
    
    def log_candidate_generated(self, candidate_number: int, model_name: str, 
                              schema_representation: str, temperature: float, 
                              generated_sql: str, is_valid: bool, 
                              validation_error: str = None, generation_time: float = None) -> None:
        sql_display = generated_sql.strip()
        if len(sql_display) > 100:
            sql_display = sql_display[:100] + "..."
        
        status = "✅ VALID" if is_valid else "❌ INVALID"
        
        self.info(f"Candidate {candidate_number} generated", phase='phase2', extra_data={
            'candidate_number': candidate_number,
            'model_name': model_name,
            'schema_representation': schema_representation,
            'temperature': temperature,
            'generated_sql': generated_sql,
            'is_valid': is_valid,
            'validation_error': validation_error,
            'generation_time': generation_time,
            'sql_candidate': True
        })
        
        print(f"\n📝 CANDIDATE {candidate_number}")
        print(f"   Model: {model_name}")
        print(f"   Schema: {schema_representation}")
        print(f"   Temperature: {temperature}")
        print(f"   Status: {status}")
        if generation_time:
            print(f"   Generation Time: {generation_time:.2f}s")
        print(f"   SQL: {sql_display}")
        if not is_valid and validation_error:
            print(f"   Error: {validation_error}")
        print(f"   {'-'*50}")
    
    def log_candidate_generation_summary(self, question_id: int, total_generated: int, 
                                       valid_count: int, invalid_count: int, 
                                       total_time: float, models_used: list) -> None:
        
        self.info(f"\n✅ CANDIDATE GENERATION COMPLETED")
        self.info(f"Total candidates: {total_generated}")
        self.info(f"Valid: {valid_count} | Invalid: {invalid_count}")
        self.info(f"Total time: {total_time:.2f}s")
        self.info(f"Models used: {', '.join(models_used)}")
        self.info(f"{'='*60}\n")
        
        print(f"\n✅ CANDIDATE GENERATION COMPLETED")
        print(f"📊 SUMMARY:")
        print(f"   Total candidates: {total_generated}")
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Invalid: {invalid_count}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   🤖 Models used: {', '.join(models_used)}")
        print(f"{'='*60}\n")
        
        self.info("Candidate generation summary", phase='phase2', extra_data={
            'question_id': question_id,
            'total_generated': total_generated,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'total_time': total_time,
            'models_used': models_used,
            'success_rate': (valid_count / total_generated * 100) if total_generated > 0 else 0,
            'generation_summary': True
        })
    
    def log_candidate_generation_summary_enhanced(self, question_id: int, total_generated: int, 
                                                valid_count: int, invalid_count: int, 
                                                fixed_count: int, total_time: float, models_used: list) -> None:
        
        self.info(f"\n✅ CANDIDATE GENERATION COMPLETED")
        self.info(f"Total candidates: {total_generated}")
        self.info(f"Valid: {valid_count} | Invalid: {invalid_count} | Fixed: {fixed_count}")
        self.info(f"Total time: {total_time:.2f}s")
        self.info(f"Models used: {', '.join(models_used)}")
        self.info(f"{'='*60}\n")
        
        print(f"\n✅ CANDIDATE GENERATION COMPLETED")
        print(f"📊 SUMMARY:")
        print(f"   Total candidates: {total_generated}")
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Invalid: {invalid_count}")
        if fixed_count > 0:
            print(f"   🔧 Fixed: {fixed_count}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   🤖 Models used: {', '.join(models_used)}")
        print(f"{'='*60}\n")
        
        self.info("Enhanced candidate generation summary", phase='phase2', extra_data={
            'question_id': question_id,
            'total_generated': total_generated,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'fixed_count': fixed_count,
            'total_time': total_time,
            'models_used': models_used,
            'success_rate': (valid_count / total_generated * 100) if total_generated > 0 else 0,
            'fix_rate': (fixed_count / invalid_count * 100) if invalid_count > 0 else 0,
            'generation_summary_enhanced': True
        })
    
    def log_schema_linking_detailed(self, question_id: int, question: str, 
                                  selected_tables: list, selected_columns: list,
                                  refinement_used: bool, refinement_reasoning: str = "",
                                  schema_representations: dict = None) -> None:
        
        self.info(f"\n🔗 SCHEMA LINKING RESULTS")
        self.info(f"Question ID: {question_id}")
        self.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        self.info(f"{'='*60}")
        
        print(f"\n🔗 SCHEMA LINKING RESULTS")
        print(f"Question ID: {question_id}")
        print(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"{'='*60}")
        
        print(f"\n📋 SELECTED TABLES ({len(selected_tables)}):")
        for i, table in enumerate(selected_tables, 1):
            print(f"   {i}. {table}")
        
        print(f"\n🏷️  SELECTED COLUMNS ({len(selected_columns)}):")
        columns_by_table = {}
        for col in selected_columns:
            if '.' in col:
                table, column = col.split('.', 1)
                if table not in columns_by_table:
                    columns_by_table[table] = []
                columns_by_table[table].append(column)
        
        for table, columns in columns_by_table.items():
            print(f"   📊 {table}:")
            for col in columns:
                print(f"      • {col}")
        
        if refinement_used:
            print(f"\n🧠 LLM REFINEMENT: ✅ Used")
            if refinement_reasoning:
                print(f"   Reasoning: {refinement_reasoning[:200]}{'...' if len(refinement_reasoning) > 200 else ''}")
        else:
            print(f"\n🧠 LLM REFINEMENT: ❌ Not used")
        
        if schema_representations:
            print(f"\n📄 SCHEMA REPRESENTATIONS:")
            for repr_type in schema_representations.keys():
                print(f"   • {repr_type}")
        
        print(f"{'='*60}\n")
        
        self.info("Schema linking detailed results", phase='phase1', extra_data={
            'question_id': question_id,
            'question': question,
            'selected_tables': selected_tables,
            'selected_columns': selected_columns,
            'tables_count': len(selected_tables),
            'columns_count': len(selected_columns),
            'refinement_used': refinement_used,
            'refinement_reasoning': refinement_reasoning,
            'schema_representations': list(schema_representations.keys()) if schema_representations else [],
            'columns_by_table': columns_by_table,
            'schema_linking_detailed': True
        })
    
    def log_api_cost(self, api_name: str, model_name: str, input_tokens: int, 
                    output_tokens: int, cost_usd: float, operation: str = "") -> None:
        
        operation_msg = f" ({operation})" if operation else ""
        
        self.info(f"💰 API Cost{operation_msg}", extra_data={
            'api_name': api_name,
            'model_name': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost_usd': cost_usd,
            'operation': operation,
            'api_cost_tracking': True
        })
        
        if 'gpt-4o' in model_name.lower() or 'gpt4o' in operation.lower():
            print(f"\n{'='*50}")
            print(f"💰 GPT-4O API COST{operation_msg}")
            print(f"{'='*50}")
            print(f"   API: {api_name}")
            print(f"   Model: {model_name}")
            print(f"   Tokens: {input_tokens} in + {output_tokens} out")
            print(f"   💵 Cost: ${cost_usd:.6f}")
            print(f"{'='*50}\n")
        else:
            print(f"💰 API Cost{operation_msg}:")
            print(f"   API: {api_name}")
            print(f"   Model: {model_name}")
            print(f"   Tokens: {input_tokens} in + {output_tokens} out")
            print(f"   Cost: ${cost_usd:.6f}")
    
    def log_total_api_costs(self, total_cost: float, cost_breakdown: dict) -> None:
        
        print(f"\n💰 TOTAL API COSTS")
        print(f"{'='*40}")
        print(f"Total: ${total_cost:.4f}")
        print(f"\nBreakdown by API:")
        
        for api_name, api_cost in cost_breakdown.items():
            print(f"   {api_name}: ${api_cost:.4f}")
        
        print(f"{'='*40}\n")
        
        self.info("Total API costs", extra_data={
            'total_cost_usd': total_cost,
            'cost_breakdown': cost_breakdown,
            'api_cost_summary': True
        })
    
    def log_sql_selection_start(self, question_id: int, total_candidates: int, 
                               selection_method: str = "") -> None:
        
        self.info(f"\n🎯 SQL SELECTION PROCESS")
        self.info(f"Question ID: {question_id}")
        self.info(f"Total candidates: {total_candidates}")
        if selection_method:
            self.info(f"Selection method: {selection_method}")
        self.info(f"{'='*60}")
        
        print(f"\n🎯 SQL SELECTION PROCESS")
        print(f"Question ID: {question_id}")
        print(f"Total candidates: {total_candidates}")
        if selection_method:
            print(f"Selection method: {selection_method}")
        print(f"{'='*60}")
    
    def log_candidate_filtering(self, original_count: int, after_syntax_filter: int, 
                              after_duplicate_filter: int, final_count: int) -> None:
        
        print(f"\n🔍 CANDIDATE FILTERING:")
        print(f"   Original candidates: {original_count}")
        if after_syntax_filter != original_count:
            print(f"   After syntax filter: {after_syntax_filter} (removed {original_count - after_syntax_filter})")
        if after_duplicate_filter != after_syntax_filter:
            print(f"   After duplicate filter: {after_duplicate_filter} (removed {after_syntax_filter - after_duplicate_filter})")
        print(f"   Final candidates for selection: {final_count}")
        print(f"   {'-'*40}")
        
        self.info("Candidate filtering results", phase='phase3', extra_data={
            'original_count': original_count,
            'after_syntax_filter': after_syntax_filter,
            'after_duplicate_filter': after_duplicate_filter,
            'final_count': final_count,
            'syntax_removed': original_count - after_syntax_filter,
            'duplicates_removed': after_syntax_filter - after_duplicate_filter,
            'candidate_filtering': True
        })
    
    def log_selected_candidate(self, question_id: int, candidate_number: int, 
                             selected_sql: str, confidence_score: float,
                             selection_method: str, model_name: str = "",
                             schema_representation: str = "", temperature: float = 0.0,
                             original_validity: bool = True, was_fixed: bool = False) -> None:
        
        self.info(f"\n🏆 SELECTED SQL CANDIDATE")
        self.info(f"Question ID: {question_id}")
        self.info(f"Candidate Number: {candidate_number}")
        self.info(f"Selection Method: {selection_method}")
        self.info(f"Confidence Score: {confidence_score:.3f}")
        
        print(f"\n🏆 SELECTED SQL CANDIDATE")
        print(f"{'='*60}")
        print(f"Question ID: {question_id}")
        print(f"Candidate Number: {candidate_number}")
        print(f"Selection Method: {selection_method}")
        print(f"Confidence Score: {confidence_score:.3f}")
        
        if model_name:
            print(f"\n📋 GENERATION CONFIGURATION:")
            print(f"   Model: {model_name}")
            if schema_representation:
                print(f"   Schema Representation: {schema_representation}")
            if temperature > 0:
                print(f"   Temperature: {temperature}")
            
            if was_fixed:
                print(f"   Original Status: ❌ Invalid (was fixed)")
                print(f"   Final Status: ✅ Valid")
            elif original_validity:
                print(f"   Status: ✅ Valid")
            else:
                print(f"   Status: ❌ Invalid")
        
        print(f"\n🔍 SELECTED SQL:")
        print(f"   {selected_sql}")
        
        print(f"{'='*60}\n")
        
        self.info("SQL candidate selected", phase='phase3', extra_data={
            'question_id': question_id,
            'candidate_number': candidate_number,
            'selected_sql': selected_sql,
            'confidence_score': confidence_score,
            'selection_method': selection_method,
            'model_name': model_name,
            'schema_representation': schema_representation,
            'temperature': temperature,
            'original_validity': original_validity,
            'was_fixed': was_fixed,
            'sql_selection_detailed': True
        })
    
    def log_selection_process_detail(self, step: str, description: str, 
                                   candidates_count: int = None) -> None:
        
        count_msg = f" ({candidates_count} candidates)" if candidates_count is not None else ""
        print(f"   🔄 {step}: {description}{count_msg}")
        
        self.info(f"Selection step: {step} - {description}", phase='phase3', extra_data={
            'selection_step': step,
            'description': description,
            'candidates_count': candidates_count,
            'selection_process_detail': True
        })
    
    def log_model_output(self, question_id: int, phase: str, model_purpose: str, 
                        model_name: str, model_type: str, prompt: str, 
                        raw_output: str, parsed_output: dict = None, 
                        output_format: str = 'text', parsing_success: bool = True) -> None:
        
        if not self.log_config.get('log_model_outputs', True):
            return
        
        display_output = raw_output[:500] + "..." if len(raw_output) > 500 else raw_output
        
        print(f"\n🤖 MODEL OUTPUT CAPTURED")
        print(f"   Question ID: {question_id}")
        print(f"   Phase: {phase}")
        print(f"   Purpose: {model_purpose}")
        print(f"   Model: {model_name} ({model_type})")
        print(f"   Format: {output_format}")
        print(f"   Parsing: {'✅ Success' if parsing_success else '❌ Failed'}")
        print(f"   Output: {display_output}")
        print(f"{'='*60}")
        
        self.info(f"Model output captured: {model_purpose}", phase=phase, extra_data={
            'question_id': question_id,
            'model_output_capture': True,
            'model_purpose': model_purpose,
            'model_name': model_name,
            'model_type': model_type,
            'output_format': output_format,
            'parsing_success': parsing_success,
            'prompt': prompt,
            'raw_output': raw_output,
            'parsed_output': parsed_output,
            'output_length': len(raw_output),
            'prompt_length': len(prompt)
        })
    
    def log_model_reuse_summary(self, model_manager) -> None:

        try:
            reuse_summary = model_manager.get_model_reuse_summary()
            
            print(f"\n♻️  MODEL REUSE EFFICIENCY SUMMARY")
            print(f"   Total models loaded: {reuse_summary['total_models_loaded']}")
            print(f"   Total purposes served: {reuse_summary['total_purposes']}")
            print(f"   Models reused for multiple purposes: {reuse_summary['models_with_multiple_purposes']}")
            print(f"   Reuse efficiency: {reuse_summary['reuse_efficiency']:.1f}%")
            
            if reuse_summary['model_details']:
                print(f"\n   📋 Model Details:")
                for model_name, details in reuse_summary['model_details'].items():
                    if details['is_reused']:
                        purposes_str = ", ".join(details['purposes'])
                        print(f"      ♻️  {model_name}: {purposes_str}")
                    else:
                        print(f"      📦 {model_name}: {details['purposes'][0]}")
            
            print(f"{'='*60}")
            
            self.info("Model reuse efficiency summary", extra_data={
                'model_reuse_summary': True,
                **reuse_summary
            })
            
        except Exception as e:
            self.warning(f"Failed to generate model reuse summary: {str(e)}")
    
    def get_model_outputs_for_question(self, question_id: int) -> List[Dict[str, Any]]:

        model_outputs = []
        for log_entry in self.json_logs:
            extra_data = log_entry.get('extra_data', {})
            if (extra_data.get('model_output_capture') and 
                extra_data.get('question_id') == question_id):
                model_outputs.append(extra_data)
        return model_outputs
    
    def get_model_outputs_by_phase(self, question_id: int, phase: str) -> List[Dict[str, Any]]:
 
        model_outputs = []
        for log_entry in self.json_logs:
            extra_data = log_entry.get('extra_data', {})
            if (extra_data.get('model_output_capture') and 
                extra_data.get('question_id') == question_id and
                log_entry.get('phase') == phase):
                model_outputs.append(extra_data)
        return model_outputs
    
    def get_model_outputs_by_purpose(self, question_id: int, model_purpose: str) -> List[Dict[str, Any]]:

        model_outputs = []
        for log_entry in self.json_logs:
            extra_data = log_entry.get('extra_data', {})
            if (extra_data.get('model_output_capture') and 
                extra_data.get('question_id') == question_id and
                extra_data.get('model_purpose') == model_purpose):
                model_outputs.append(extra_data)
        return model_outputs
    
    def get_all_model_outputs(self) -> List[Dict[str, Any]]:

        model_outputs = []
        for log_entry in self.json_logs:
            extra_data = log_entry.get('extra_data', {})
            if extra_data.get('model_output_capture'):
                model_outputs.append(extra_data)
        return model_outputs
    
    def get_model_outputs_summary_for_question(self, question_id: int) -> Dict[str, Any]:

        outputs = self.get_model_outputs_for_question(question_id)
        
        by_phase = defaultdict(list)
        by_purpose = defaultdict(list)
        by_model = defaultdict(list)
        
        for output in outputs:
            phase = output.get('phase', 'unknown')
            purpose = output.get('model_purpose', 'unknown')
            model_name = output.get('model_name', 'unknown')
            
            by_phase[phase].append(output)
            by_purpose[purpose].append(output)
            by_model[model_name].append(output)
        
        return {
            'question_id': question_id,
            'total_model_outputs': len(outputs),
            'outputs_by_phase': {phase: len(outputs) for phase, outputs in by_phase.items()},
            'outputs_by_purpose': {purpose: len(outputs) for purpose, outputs in by_purpose.items()},
            'outputs_by_model': {model: len(outputs) for model, outputs in by_model.items()},
            'all_outputs': outputs
        }

    def close(self) -> None:

        self.save_json_logs()
        self.info("Pipeline logger closed")
        
        for handler in self.main_logger.handlers:
            handler.close()
        
        for logger in self.phase_loggers.values():
            for handler in logger.handlers:
                handler.close()
