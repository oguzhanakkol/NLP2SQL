import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict


@dataclass
class QuestionStatistics:
    question_id: int
    db_id: str
    difficulty: str
    
    total_time: float = 0.0
    phase1_time: float = 0.0
    phase2_time: float = 0.0
    phase3_time: float = 0.0
    evaluation_time: float = 0.0
    
    selected_tables_count: int = 0
    selected_columns_count: int = 0
    refinement_used: bool = False
    
    candidates_generated: int = 0
    valid_candidates: int = 0
    invalid_candidates: int = 0
    models_used: List[str] = None
    
    selection_method: str = ""
    final_sql_valid: bool = False
    execution_successful: bool = False
    
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    
    correct_execution: bool = False
    
    def __post_init__(self):
        if self.models_used is None:
            self.models_used = []


@dataclass
class PhaseStatistics:
    phase_name: str
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0
    
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class StatisticsTracker:
    
    def __init__(self, config_manager):

        self.config = config_manager
        self.start_time = datetime.now().isoformat()
        
        self.question_stats: Dict[int, QuestionStatistics] = {}
        
        self.phase_stats: Dict[str, PhaseStatistics] = {
            'phase1_schema_linking': PhaseStatistics('phase1_schema_linking'),
            'phase2_sql_generation': PhaseStatistics('phase2_sql_generation'),
            'phase3_sql_selection': PhaseStatistics('phase3_sql_selection'),
            'evaluation': PhaseStatistics('evaluation')
        }
        
        self.model_usage: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': 0.0,
            'avg_latency': 0.0,
            'latencies': []
        })
        
        self.errors: List[Dict[str, Any]] = []
        
        self.performance_metrics = {
            'memory_usage': [],
            'gpu_memory_usage': [],
            'processing_speed': []
        }
        
        self.config_snapshot = config_manager.config.copy()
    
    def start_question_processing(self, question_id: int, db_id: str, difficulty: str) -> None:

        self.question_stats[question_id] = QuestionStatistics(
            question_id=question_id,
            db_id=db_id,
            difficulty=difficulty
        )
    
    def log_phase_timing(self, question_id: int, phase: str, duration: float) -> None:

        if question_id not in self.question_stats:
            return
        
        stats = self.question_stats[question_id]
        
        if phase == 'phase1':
            stats.phase1_time = duration
        elif phase == 'phase2':
            stats.phase2_time = duration
        elif phase == 'phase3':
            stats.phase3_time = duration
        elif phase == 'evaluation':
            stats.evaluation_time = duration
        
        stats.total_time = (stats.phase1_time + stats.phase2_time + 
                           stats.phase3_time + stats.evaluation_time)
        
        phase_key = f"{phase}_{'schema_linking' if phase == 'phase1' else 'sql_generation' if phase == 'phase2' else 'sql_selection' if phase == 'phase3' else phase}"
        if phase_key in self.phase_stats:
            phase_stat = self.phase_stats[phase_key]
            phase_stat.total_processing_time += duration
            if phase_stat.total_questions > 0:
                phase_stat.avg_processing_time = phase_stat.total_processing_time / phase_stat.total_questions
    
    def log_schema_linking_result(self, question_id: int, selected_tables: int, 
                                 selected_columns: int, refinement_used: bool) -> None:

        if question_id not in self.question_stats:
            return
        
        stats = self.question_stats[question_id]
        stats.selected_tables_count = selected_tables
        stats.selected_columns_count = selected_columns
        stats.refinement_used = refinement_used
    
    def log_sql_generation_result(self, question_id: int, candidates_generated: int,
                                 valid_candidates: int, models_used: List[str]) -> None:

        if question_id not in self.question_stats:
            return
        
        stats = self.question_stats[question_id]
        stats.candidates_generated = candidates_generated
        stats.valid_candidates = valid_candidates
        stats.invalid_candidates = candidates_generated - valid_candidates
        stats.models_used = models_used.copy()
    
    def log_sql_selection_result(self, question_id: int, selection_method: str,
                                final_sql_valid: bool) -> None:
 
        if question_id not in self.question_stats:
            return
        
        stats = self.question_stats[question_id]
        stats.selection_method = selection_method
        stats.final_sql_valid = final_sql_valid
    
    def log_execution_result(self, question_id: int, execution_successful: bool,
                            correct_execution: bool) -> None:
 
        if question_id not in self.question_stats:
            return
        
        stats = self.question_stats[question_id]
        stats.execution_successful = execution_successful
        stats.correct_execution = correct_execution
    
    def log_token_usage(self, question_id: int, model_name: str, input_tokens: int,
                       output_tokens: int, cost: Optional[float] = None) -> None:

        if question_id in self.question_stats:
            stats = self.question_stats[question_id]
            stats.total_input_tokens += input_tokens
            stats.total_output_tokens += output_tokens
            if cost:
                stats.total_cost += cost
        
        model_stats = self.model_usage[model_name]
        model_stats['calls'] += 1
        model_stats['input_tokens'] += input_tokens
        model_stats['output_tokens'] += output_tokens
        if cost:
            model_stats['cost'] += cost
    
    def log_model_latency(self, model_name: str, latency: float) -> None:

        model_stats = self.model_usage[model_name]
        model_stats['latencies'].append(latency)
        model_stats['avg_latency'] = sum(model_stats['latencies']) / len(model_stats['latencies'])
    
    def log_error(self, question_id: Optional[int], phase: str, error_type: str,
                  error_message: str, **context) -> None:

        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'question_id': question_id,
            'phase': phase,
            'error_type': error_type,
            'error_message': error_message,
            'context': context
        }
        self.errors.append(error_entry)
        
        phase_key = f"{phase}_{'schema_linking' if phase == 'phase1' else 'sql_generation' if phase == 'phase2' else 'sql_selection' if phase == 'phase3' else phase}"
        if phase_key in self.phase_stats:
            self.phase_stats[phase_key].failed_questions += 1
    
    def log_phase_success(self, phase: str) -> None:

        phase_key = f"{phase}_{'schema_linking' if phase == 'phase1' else 'sql_generation' if phase == 'phase2' else 'sql_selection' if phase == 'phase3' else phase}"
        if phase_key in self.phase_stats:
            phase_stat = self.phase_stats[phase_key]
            phase_stat.successful_questions += 1
            phase_stat.total_questions += 1
    
    def log_question_statistics(self, question_result: Dict[str, Any]) -> None:

        question_id = question_result['question_id']
        
        if 'schema_linking' in question_result:
            schema_result = question_result['schema_linking']
            self.log_schema_linking_result(
                question_id,
                len(schema_result.get('selected_tables', [])),
                len(schema_result.get('selected_columns', [])),
                schema_result.get('refinement_used', False)
            )
        
        if 'sql_candidates' in question_result:
            candidates_result = question_result['sql_candidates']
            self.log_sql_generation_result(
                question_id,
                len(candidates_result.get('candidates', [])),
                len([c for c in candidates_result.get('candidates', []) if c.get('valid', False)]),
                candidates_result.get('models_used', [])
            )
        
        if 'final_sql' in question_result:
            final_result = question_result['final_sql']
            selected_sql = final_result.get('selected_sql', '').strip()
            has_sql = bool(selected_sql)
            
            self.log_sql_selection_result(
                question_id,
                final_result.get('selection_method', 'unknown'),
                has_sql
            )
    
    def get_progressive_metrics(self) -> Dict[str, Any]:

        if not self.question_stats:
            return {
                'questions_processed': 0,
                'accuracy_rate': 0.0,
                'execution_success_rate': 0.0,
                'sql_generation_success_rate': 0.0,
                'questions_with_ground_truth': 0
            }
        
        total_questions = len(self.question_stats)
        questions_with_sql = sum(1 for stats in self.question_stats.values() if stats.final_sql_valid)
        successful_executions = sum(1 for stats in self.question_stats.values() if stats.execution_successful)
        correct_executions = sum(1 for stats in self.question_stats.values() if stats.correct_execution)
        questions_with_ground_truth = sum(1 for stats in self.question_stats.values() if stats.correct_execution or stats.execution_successful)
        
        return {
            'questions_processed': total_questions,
            'sql_generation_success_rate': (questions_with_sql / total_questions * 100) if total_questions > 0 else 0.0,
            'execution_success_rate': (successful_executions / total_questions * 100) if total_questions > 0 else 0.0,
            'accuracy_rate': (correct_executions / questions_with_ground_truth * 100) if questions_with_ground_truth > 0 else 0.0,
            'questions_with_ground_truth': questions_with_ground_truth,
            'breakdown': {
                'total_processed': total_questions,
                'sql_generated': questions_with_sql,
                'execution_successful': successful_executions,
                'correct_results': correct_executions,
                'failed_generation': total_questions - questions_with_sql,
                'failed_execution': total_questions - successful_executions,
                'incorrect_results': successful_executions - correct_executions
            }
        }
    
    def print_progressive_metrics(self, logger=None) -> None:

        metrics = self.get_progressive_metrics()
        
        output_lines = [
            "",
            "📊 PROGRESSIVE PERFORMANCE METRICS",
            "=" * 50,
            f"Questions Processed: {metrics['questions_processed']}",
            f"SQL Generation Success: {metrics['sql_generation_success_rate']:.1f}% ({metrics['breakdown']['sql_generated']}/{metrics['breakdown']['total_processed']})",
            f"Execution Success Rate: {metrics['execution_success_rate']:.1f}% ({metrics['breakdown']['execution_successful']}/{metrics['breakdown']['total_processed']})",
            f"Answer Accuracy Rate: {metrics['accuracy_rate']:.1f}% ({metrics['breakdown']['correct_results']}/{metrics['questions_with_ground_truth']})",
            "",
            "📈 SIMPLIFIED BREAKDOWN:",
            f"  ✅ Successful Executions: {metrics['breakdown']['execution_successful']}",
            f"  ✅ Correct Results: {metrics['breakdown']['correct_results']}",
            f"  ❌ Failed Executions: {metrics['breakdown']['failed_execution']}",
            f"  ❌ Incorrect Results: {metrics['breakdown']['incorrect_results']}",
            "=" * 50
        ]
        
        output = "\n".join(output_lines)
        
        if logger:
            for line in output_lines:
                if line.strip():
                    logger.info(line)
        else:
            print(output)
    
    def get_summary(self) -> Dict[str, Any]:

        total_questions = len(self.question_stats)
        
        if total_questions == 0:
            return {
                'overview': {
                    'total_questions': 0,
                    'pipeline_status': 'No questions processed'
                }
            }
        
        correct_executions = sum(1 for stats in self.question_stats.values() if stats.correct_execution)
        successful_executions = sum(1 for stats in self.question_stats.values() if stats.execution_successful)
        
        total_cost = sum(stats.total_cost for stats in self.question_stats.values())
        total_time = sum(stats.total_time for stats in self.question_stats.values())
        
        difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for stats in self.question_stats.values():
            difficulty_stats[stats.difficulty]['total'] += 1
            if stats.correct_execution:
                difficulty_stats[stats.difficulty]['correct'] += 1
        
        db_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for stats in self.question_stats.values():
            db_stats[stats.db_id]['total'] += 1
            if stats.correct_execution:
                db_stats[stats.db_id]['correct'] += 1
        
        model_summary = {}
        for model_name, usage in self.model_usage.items():
            model_summary[model_name] = {
                'total_calls': usage['calls'],
                'total_tokens': usage['input_tokens'] + usage['output_tokens'],
                'total_cost': usage['cost'],
                'avg_latency': usage['avg_latency']
            }
        
        return {
            'overview': {
                'total_questions': total_questions,
                'correct_executions': correct_executions,
                'execution_accuracy': correct_executions / total_questions * 100,
                'successful_executions': successful_executions,
                'execution_success_rate': successful_executions / total_questions * 100,
                'total_cost': total_cost,
                'total_processing_time': total_time,
                'avg_processing_time': total_time / total_questions,
                'start_time': self.start_time,
                'end_time': datetime.now().isoformat()
            },
            'difficulty_breakdown': {
                difficulty: {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                }
                for difficulty, stats in difficulty_stats.items()
            },
            'database_breakdown': {
                db_id: {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                }
                for db_id, stats in db_stats.items()
            },
            'phase_statistics': {
                phase_name: asdict(phase_stats)
                for phase_name, phase_stats in self.phase_stats.items()
            },
            'model_usage': model_summary,
            'error_summary': {
                'total_errors': len(self.errors),
                'errors_by_phase': dict(Counter(error['phase'] for error in self.errors)),
                'errors_by_type': dict(Counter(error['error_type'] for error in self.errors))
            },
            'token_usage': {
                'total_input_tokens': sum(stats.total_input_tokens for stats in self.question_stats.values()),
                'total_output_tokens': sum(stats.total_output_tokens for stats in self.question_stats.values()),
                'avg_tokens_per_question': {
                    'input': sum(stats.total_input_tokens for stats in self.question_stats.values()) / total_questions,
                    'output': sum(stats.total_output_tokens for stats in self.question_stats.values()) / total_questions
                }
            }
        }
    
    def export_detailed_statistics(self, output_path: str) -> None:

        detailed_stats = {
            'summary': self.get_summary(),
            'question_level_statistics': {
                str(qid): asdict(stats) for qid, stats in self.question_stats.items()
            },
            'detailed_model_usage': dict(self.model_usage),
            'errors': self.errors,
            'configuration_snapshot': self.config_snapshot
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(detailed_stats, f, indent=2, default=str)
    
    def print_summary_report(self) -> None:

        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("NATURAL LANGUAGE TO SQL PIPELINE - STATISTICS SUMMARY")
        print("="*80)
        
        overview = summary['overview']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Questions Processed: {overview['total_questions']}")
        print(f"  Execution Accuracy: {overview['execution_accuracy']:.2f}%")
        print(f"  Success Rate: {overview['execution_success_rate']:.2f}%")
        print(f"  Total Cost: ${overview['total_cost']:.4f}")
        print(f"  Average Processing Time: {overview['avg_processing_time']:.2f}s per question")
        
        print(f"\nDIFFICULTY BREAKDOWN:")
        for difficulty, stats in summary['difficulty_breakdown'].items():
            print(f"  {difficulty.capitalize()}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2f}%)")
        
        print(f"\nMODEL USAGE:")
        for model_name, usage in summary['model_usage'].items():
            print(f"  {model_name}: {usage['total_calls']} calls, ${usage['total_cost']:.4f}")
        
        if summary['error_summary']['total_errors'] > 0:
            print(f"\nERRORS:")
            print(f"  Total Errors: {summary['error_summary']['total_errors']}")
            for phase, count in summary['error_summary']['errors_by_phase'].items():
                print(f"    {phase}: {count} errors")
        
        print("\n" + "="*80)
    
    def get_real_time_metrics(self) -> Dict[str, Any]:

        if not self.question_stats:
            return {'status': 'No questions processed yet'}
        
        processed_questions = len(self.question_stats)
        recent_questions = list(self.question_stats.values())[-10:]
        
        recent_accuracy = sum(1 for q in recent_questions if q.correct_execution) / len(recent_questions) * 100
        recent_avg_time = sum(q.total_time for q in recent_questions) / len(recent_questions)
        
        return {
            'processed_questions': processed_questions,
            'recent_accuracy': recent_accuracy,
            'recent_avg_time': recent_avg_time,
            'total_cost_so_far': sum(q.total_cost for q in self.question_stats.values()),
            'current_processing_rate': processed_questions / ((datetime.now() - datetime.fromisoformat(self.start_time)).total_seconds() / 3600),
            'recent_errors': len([e for e in self.errors[-10:] if e])
        }
