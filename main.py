import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config_manager import ConfigManager
from src.core.logger import PipelineLogger
from src.core.data_loader import BirdDataLoader
from src.core.checkpoint_manager import CheckpointManager
from src.core.model_manager import ModelManager
from src.core.statistics_tracker import StatisticsTracker

from src.phases.phase1_schema_linking.schema_linker import SchemaLinker
from src.phases.phase2_sql_generation.sql_generator import SQLGenerator
from src.phases.phase3_sql_selection.sql_selector import SQLSelector

from tools.evaluation.bird_evaluator import BirdEvaluator


class NLPToSQLPipeline:
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = PipelineLogger(self.config)
        self.data_loader = BirdDataLoader(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.model_manager.set_logger(self.logger)
        self.statistics_tracker = StatisticsTracker(self.config)
        
        self.schema_linker = SchemaLinker(self.config, self.model_manager)
        self.schema_linker.set_logger(self.logger)
        self.sql_generator = SQLGenerator(self.config, self.model_manager)
        self.sql_generator.set_logger(self.logger)
        self.sql_selector = SQLSelector(self.config, self.model_manager)
        self.sql_selector.set_logger(self.logger)
        self.evaluator = BirdEvaluator(self.config)
        
        self.logger.info("Pipeline initialized successfully")
        
        include_outputs = self.config.get('execution.include_model_outputs_in_results', True)
        log_outputs = self.config.get('logging.log_model_outputs', True)
        
        if include_outputs:
            self.logger.info("✅ Model raw outputs will be included in pipeline results JSON")
        else:
            self.logger.info("❌ Model raw outputs will NOT be included in pipeline results JSON")
            
        if log_outputs:
            self.logger.info("✅ Model outputs logging is ENABLED")
        else:
            self.logger.info("❌ Model outputs logging is DISABLED")
            
        if hasattr(self.logger, 'json_logs'):
            self.logger.info(f"✅ Logger has json_logs attribute (currently {len(self.logger.json_logs or [])} entries)")
        else:
            self.logger.info("❌ Logger does NOT have json_logs attribute")
    
    def run_full_pipeline(self, start_from_checkpoint: bool = False) -> Dict:
        try:
            self.logger.info("Starting Natural Language to SQL Pipeline")
            
            questions = self.data_loader.load_questions()
            databases = self.data_loader.load_databases()
            
            start_idx = 0
            if start_from_checkpoint:
                checkpoint = self.checkpoint_manager.load_latest_checkpoint()
                if checkpoint:
                    start_idx = checkpoint.get('last_processed_question', 0) + 1
                    self.logger.info(f"Resuming from checkpoint at question {start_idx}")
            
            results = []
            max_questions = self.config.get('execution.max_questions')
            questions_to_process = questions[start_idx:max_questions] if max_questions else questions[start_idx:]
            
            for i, question in enumerate(questions_to_process, start=start_idx):
                self.logger.info(f"Processing question {i+1}/{len(questions)}: {question['question_id']}")
                
                try:
                    self.statistics_tracker.start_question_processing(
                        question['question_id'], 
                        question['db_id'], 
                        question.get('difficulty', 'unknown')
                    )
                    
                    schema_result = self._run_phase1(question, databases)
                    
                    candidates_result = self._run_phase2(question, schema_result)
                    
                    final_sql_result = self._run_phase3(question, candidates_result)
                    
                    question_result = {
                        'question_id': question['question_id'],
                        'db_id': question['db_id'],
                        'question': question['question'],
                        'ground_truth_sql': question['SQL'],
                        'difficulty': question['difficulty'],
                        'schema_linking': schema_result,
                        'sql_candidates': candidates_result,
                        'final_sql': final_sql_result
                    }
                    
                    if self.config.get('execution.include_model_outputs_in_results', True):
                        model_outputs = self._get_model_outputs_for_question(question['question_id'])
                        if model_outputs:
                            question_result['model_outputs'] = model_outputs
                            self.logger.debug(f"📋 Collected {model_outputs['total_model_calls']} model outputs for question {question['question_id']}")
                        else:
                            self.logger.debug(f"⚠️ No model outputs found for question {question['question_id']}")
                    
                    evaluation_result = self._evaluate_question_progressively(question_result, i + 1, len(questions_to_process) + start_idx)
                    
                    if evaluation_result:
                        question_result['sql_execution'] = {
                            'predicted_sql_execution': {
                                'sql': evaluation_result.get('predicted_sql', question_result.get('final_sql', {}).get('selected_sql', '')),
                                'execution_success': evaluation_result.get('execution_success', False),
                                'execution_result': evaluation_result.get('predicted_execution_result'),
                                'execution_error': evaluation_result.get('execution_error'),
                                'result_count': len(evaluation_result.get('predicted_execution_result', []))
                            },
                            'ground_truth_sql_execution': {
                                'sql': evaluation_result.get('ground_truth_sql', question['SQL']),
                                'execution_success': evaluation_result.get('ground_truth_execution_result') is not None,
                                'execution_result': evaluation_result.get('ground_truth_execution_result'),
                                'execution_error': evaluation_result.get('ground_truth_execution_error'),
                                'result_count': len(evaluation_result.get('ground_truth_execution_result', []))
                            },
                            'comparison': {
                                'is_correct': evaluation_result.get('is_correct', False),
                                'ground_truth_available': evaluation_result.get('ground_truth_available', False),
                                'results_match': evaluation_result.get('is_correct', False)
                            }
                        }
                    
                    results.append(question_result)
                    
                    if (i + 1) % self.config.get('phase2_sql_generation.checkpoint_interval', 100) == 0:
                        self.checkpoint_manager.save_checkpoint({
                            'last_processed_question': i,
                            'results': results,
                            'timestamp': datetime.now().isoformat()
                        })
                        self.logger.info(f"Checkpoint saved at question {i+1}")
                    
                    self.statistics_tracker.log_question_statistics(question_result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing question {question['question_id']}: {str(e)}")
                    if not self.config.get('advanced.continue_on_error', True):
                        raise
            
            self.logger.info("Running final evaluation")
            evaluation_results = self.evaluator.evaluate_results(results)
            
            self._log_total_costs()
            
            if hasattr(self.logger, 'json_logs') and self.logger.json_logs:
                total_model_outputs = sum(1 for entry in self.logger.json_logs 
                                        if isinstance(entry, dict) and 
                                        entry.get('extra_data', {}).get('model_output_capture'))
                self.logger.info(f"🔍 Session captured {total_model_outputs} total model outputs")
            else:
                self.logger.info("🔍 No model outputs found in session logs")
            
            final_results = {
                'pipeline_results': results,
                'evaluation': evaluation_results,
                'statistics': self.statistics_tracker.get_summary(),
                'execution_info': {
                    'total_questions': len(results),
                    'start_time': self.statistics_tracker.start_time,
                    'end_time': datetime.now().isoformat(),
                    'config_used': self.config.config
                }
            }
            
            self._save_final_results(final_results)
            
            self.logger.info("Pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def run_demo(self, question_id: int = 0) -> Dict:

        self.logger.info(f"Running pipeline demo with question {question_id}")
        

        questions = self.data_loader.load_questions()
        databases = self.data_loader.load_databases()
        
        question = next((q for q in questions if q['question_id'] == question_id), None)
        if not question:
            raise ValueError(f"Question {question_id} not found")
        
        self.logger.info(f"Demo question: {question['question']}")
        self.logger.info(f"Database: {question['db_id']}")
        self.logger.info(f"Difficulty: {question['difficulty']}")
        
        schema_result = self._run_phase1(question, databases)
        candidates_result = self._run_phase2(question, schema_result)
        final_sql_result = self._run_phase3(question, candidates_result)
        
        demo_result = {
            'question': question,
            'schema_linking': schema_result,
            'sql_candidates': candidates_result,
            'final_sql': final_sql_result
        }
        
        demo_path = Path(self.config.get('data.results_path')) / 'demo_results.json'
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        with open(demo_path, 'w') as f:
            json.dump(demo_result, f, indent=2)
        
        self.logger.info(f"Demo completed. Results saved to {demo_path}")
        return demo_result
    
    def _run_phase1(self, question: Dict, databases: Dict) -> Dict:
        start_time = time.time()
        self.logger.log_phase_start_detailed("PHASE 1: SCHEMA LINKING", question['question_id'], question['question'])
        
        try:
            result = self.schema_linker.link_schema(question, databases[question['db_id']])
            duration = time.time() - start_time
            
            self.logger.log_schema_linking_detailed(
                question['question_id'],
                question['question'],
                result['selected_tables'],
                result['selected_columns'],
                result.get('refinement_used', False),
                result.get('refinement_reasoning', ''),
                result.get('schema_representations', {})
            )
            
            summary = f"Selected {len(result['selected_tables'])} tables, {len(result['selected_columns'])} columns"
            self.logger.log_phase_end_detailed("PHASE 1: SCHEMA LINKING", question['question_id'], duration, True, summary)
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_phase_end_detailed("PHASE 1: SCHEMA LINKING", question['question_id'], duration, False, str(e))
            raise
    
    def _run_phase2(self, question: Dict, schema_result: Dict) -> Dict:
        start_time = time.time()
        self.logger.log_phase_start_detailed("PHASE 2: SQL GENERATION", question['question_id'], question['question'])
        
        try:
            result = self.sql_generator.generate_candidates(question, schema_result)
            duration = time.time() - start_time
            
            total_candidates = len(result.get('candidates', []))
            valid_candidates = len([c for c in result.get('candidates', []) if c.get('is_valid', False)])
            summary = f"Generated {total_candidates} candidates ({valid_candidates} valid)"
            self.logger.log_phase_end_detailed("PHASE 2: SQL GENERATION", question['question_id'], duration, True, summary)
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_phase_end_detailed("PHASE 2: SQL GENERATION", question['question_id'], duration, False, str(e))
            raise
    
    def _run_phase3(self, question: Dict, candidates_result: Dict) -> Dict:
        start_time = time.time()
        self.logger.log_phase_start_detailed("PHASE 3: SQL SELECTION", question['question_id'], question['question'])
        
        try:
            result = self.sql_selector.select_best_sql(question, candidates_result)
            duration = time.time() - start_time
            
            method = result.get('selection_method', 'unknown')
            confidence = result.get('confidence_score', 0)
            summary = f"Selected SQL using {method} (confidence: {confidence:.3f})"
            self.logger.log_phase_end_detailed("PHASE 3: SQL SELECTION", question['question_id'], duration, True, summary)
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_phase_end_detailed("PHASE 3: SQL SELECTION", question['question_id'], duration, False, str(e))
            raise
    
    def _save_final_results(self, results: Dict) -> None:
        results_dir = Path(self.config.get('data.results_path'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_path = results_dir / f'pipeline_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        eval_path = results_dir / f'evaluation_summary_{timestamp}.json'
        with open(eval_path, 'w') as f:
            json.dump(results['evaluation'], f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
        self.logger.info(f"Evaluation summary saved to {eval_path}")
    
    def get_model_outputs_for_question(self, question_id: int) -> List[Dict]:

        return self.logger.get_model_outputs_for_question(question_id)
    
    def get_model_outputs_by_phase(self, question_id: int, phase: str) -> List[Dict]:

        return self.logger.get_model_outputs_by_phase(question_id, phase)
    
    def get_model_outputs_by_purpose(self, question_id: int, model_purpose: str) -> List[Dict]:

        return self.logger.get_model_outputs_by_purpose(question_id, model_purpose)

    def _log_total_costs(self) -> None:
        model_usage = self.model_manager.get_model_usage_summary()
        
        cost_breakdown = {}
        total_cost = 0.0
        
        for model_key, usage_data in model_usage['usage_by_model'].items():
            cost = usage_data.get('cost', 0.0)
            total_cost += cost
            
            if 'gpt' in model_key.lower() or 'openai' in model_key.lower():
                cost_breakdown['OpenAI'] = cost_breakdown.get('OpenAI', 0.0) + cost
            elif 'gemini' in model_key.lower() or 'google' in model_key.lower():
                cost_breakdown['Google Gemini'] = cost_breakdown.get('Google Gemini', 0.0) + cost
        
        if total_cost > 0:
            self.logger.log_total_api_costs(total_cost, cost_breakdown)
    
    def _evaluate_question_progressively(self, question_result: Dict, current_question: int, total_questions: int) -> Optional[Dict]:

        try:
            eval_result = self.evaluator.evaluate_single_question(question_result)
            
            question_id = question_result['question_id']
            self.statistics_tracker.log_execution_result(
                question_id,
                eval_result.get('execution_success', False),
                eval_result.get('is_correct', False)
            )
            
            if self.config.get('execution.log_sql_execution_to_json', True):
                predicted_sql = question_result.get('final_sql', {}).get('selected_sql', '')
                if predicted_sql.strip():
                    self.logger.log_sql_execution_results(
                        question_id=question_id,
                        predicted_sql=predicted_sql,
                        predicted_result=eval_result.get('predicted_execution_result'),
                        predicted_error=eval_result.get('execution_error'),
                        ground_truth_sql=eval_result.get('ground_truth_sql'),
                        ground_truth_result=eval_result.get('ground_truth_execution_result'),
                        ground_truth_error=eval_result.get('ground_truth_execution_error'),
                        is_correct=eval_result.get('is_correct'),
                        execution_success=eval_result.get('execution_success', False)
                    )
            
            show_metrics = self.config.get('execution.show_progressive_metrics', True)
            frequency = self.config.get('execution.progressive_metrics_frequency', 1)
            
            should_display = (show_metrics and 
                            (current_question % frequency == 0 or 
                             current_question == total_questions or
                             current_question <= 3))
            
            if should_display:
                self.logger.info("")
                self.logger.info("=" * 60)
                self.logger.info(f"📊 PROGRESSIVE METRICS - Question {current_question}/{total_questions}")
                self.logger.info("=" * 60)
                
                self.statistics_tracker.print_progressive_metrics(self.logger)
                
                self.logger.info("")
                self.logger.info(f"🔍 CURRENT QUESTION ({question_result['question_id']}):")
                
                has_sql = bool(question_result.get('final_sql', {}).get('selected_sql', '').strip())
                self.logger.info(f"  SQL Generated: {'✅ Yes' if has_sql else '❌ No'}")
                
                if has_sql:
                    self.logger.info(f"  Execution Success: {'✅ Yes' if eval_result.get('execution_success') else '❌ No'}")
                    if eval_result.get('execution_error'):
                        self.logger.info(f"  Execution Error: {eval_result['execution_error']}")
                    
                    if eval_result.get('ground_truth_available'):
                        self.logger.info(f"  Answer Correct: {'✅ Yes' if eval_result.get('is_correct') else '❌ No'}")
                        if eval_result.get('ground_truth_execution_error'):
                            self.logger.info(f"  Ground Truth Error: {eval_result['ground_truth_execution_error']}")
                    else:
                        self.logger.info(f"  Answer Correct: ❓ No ground truth available")
                        
                    generated_sql = question_result.get('final_sql', {}).get('selected_sql', '')
                    if len(generated_sql) > 100:
                        sql_preview = generated_sql[:100] + "..."
                    else:
                        sql_preview = generated_sql
                    self.logger.info(f"  Generated SQL: {sql_preview}")
                    
                    show_execution_results = self.config.get('execution.show_sql_execution_results', True)
                    max_rows_to_show = self.config.get('execution.max_result_rows_display', 3)
                    
                    if (show_execution_results and 
                        eval_result.get('execution_success') and 
                        eval_result.get('predicted_execution_result') is not None):
                        
                        predicted_result = eval_result['predicted_execution_result']
                        result_count = len(predicted_result)
                        self.logger.info(f"  📋 Predicted Result: {result_count} rows")
                        
                        if result_count > 0 and max_rows_to_show > 0:
                            for i, row in enumerate(predicted_result[:max_rows_to_show]):
                                row_str = str(row)
                                if len(row_str) > 80:
                                    row_str = row_str[:80] + "..."
                                self.logger.info(f"    Row {i+1}: {row_str}")
                            if result_count > max_rows_to_show:
                                self.logger.info(f"    ... and {result_count - max_rows_to_show} more rows")
                        
                        if (eval_result.get('ground_truth_available') and 
                            eval_result.get('ground_truth_execution_result') is not None):
                            gt_result = eval_result['ground_truth_execution_result']
                            gt_count = len(gt_result)
                            self.logger.info(f"  🎯 Ground Truth Result: {gt_count} rows")
                            
                            if gt_count > 0 and max_rows_to_show > 0:
                                for i, row in enumerate(gt_result[:max_rows_to_show]):
                                    row_str = str(row)
                                    if len(row_str) > 80:
                                        row_str = row_str[:80] + "..."
                                    self.logger.info(f"    Row {i+1}: {row_str}")
                                if gt_count > max_rows_to_show:
                                    self.logger.info(f"    ... and {gt_count - max_rows_to_show} more rows")
                            
                            if eval_result.get('is_correct'):
                                self.logger.info(f"  ✅ Results match ground truth!")
                            else:
                                self.logger.info(f"  ❌ Results differ from ground truth")
                                if result_count != gt_count:
                                    self.logger.info(f"     Row count mismatch: {result_count} vs {gt_count}")
                
                self.logger.info("=" * 60)
                self.logger.info("")
            
            return eval_result
                
        except Exception as e:
            self.logger.error(f"Progressive evaluation failed for question {question_result['question_id']}: {str(e)}")
            return None
    
    def _get_model_outputs_for_question(self, question_id: int) -> Dict[str, Any]:

        try:
            if not hasattr(self.logger, 'json_logs'):
                self.logger.debug(f"⚠️ Logger does not have json_logs attribute")
                return {}
            
            total_log_entries = len(self.logger.json_logs) if self.logger.json_logs else 0
            self.logger.debug(f"🔍 Searching {total_log_entries} log entries for question {question_id}")
            
            model_outputs = []
            model_output_count = 0
            for log_entry in self.logger.json_logs:
                if isinstance(log_entry, dict):
                    extra_data = log_entry.get('extra_data', {})
                    if extra_data.get('model_output_capture'):
                        model_output_count += 1
                        if extra_data.get('question_id') == question_id:
                            
                            output_data = {
                                'phase': log_entry.get('phase'),
                                'model_purpose': extra_data.get('model_purpose'),
                                'model_name': extra_data.get('model_name'),
                                'model_type': extra_data.get('model_type'),
                                'prompt': extra_data.get('prompt'),
                                'raw_output': extra_data.get('raw_output'),
                                'parsed_output': extra_data.get('parsed_output'),
                                'output_format': extra_data.get('output_format'),
                                'parsing_success': extra_data.get('parsing_success'),
                                'output_length': extra_data.get('output_length'),
                                'prompt_length': extra_data.get('prompt_length'),
                                'timestamp': log_entry.get('timestamp')
                            }
                            model_outputs.append(output_data)
            
            self.logger.debug(f"🔍 Found {model_output_count} total model outputs, {len(model_outputs)} for question {question_id}")
            
            organized_outputs = {
                'total_model_calls': len(model_outputs),
                'by_phase': {},
                'by_purpose': {},
                'chronological_outputs': model_outputs
            }
            
            for output in model_outputs:
                phase = output.get('phase', 'unknown')
                if phase not in organized_outputs['by_phase']:
                    organized_outputs['by_phase'][phase] = []
                organized_outputs['by_phase'][phase].append(output)
            
            for output in model_outputs:
                purpose = output.get('model_purpose', 'unknown')
                if purpose not in organized_outputs['by_purpose']:
                    organized_outputs['by_purpose'][purpose] = []
                organized_outputs['by_purpose'][purpose].append(output)
            
            return organized_outputs
            
        except Exception as e:
            self.logger.error(f"Failed to get model outputs for question {question_id}: {str(e)}")
            return {}


def main():
    parser = argparse.ArgumentParser(description='Natural Language to SQL Pipeline')
    parser.add_argument('--config', default='configs/pipeline_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode with single question')
    parser.add_argument('--demo-question-id', type=int, default=0,
                        help='Question ID for demo mode')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--max-questions', type=int,
                        help='Maximum number of questions to process')
    
    args = parser.parse_args()
    
    pipeline = NLPToSQLPipeline(args.config)
    
    if args.max_questions:
        pipeline.config.config['execution']['max_questions'] = args.max_questions
    
    try:
        if args.demo:
            results = pipeline.run_demo(args.demo_question_id)
            print("Demo completed successfully!")
            print(f"Final SQL: {results['final_sql']['selected_sql']}")
        else:
            results = pipeline.run_full_pipeline(start_from_checkpoint=args.resume)
            print("Pipeline completed successfully!")
            print(f"Processed {len(results['pipeline_results'])} questions")
            print(f"Overall accuracy: {results['evaluation']['overall_accuracy']:.2f}%")
    
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
