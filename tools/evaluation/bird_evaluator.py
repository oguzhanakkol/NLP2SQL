import json
import sqlite3
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from func_timeout import func_timeout, FunctionTimedOut

class BirdEvaluator:
    
    def __init__(self, config):

        self.config = config
        self.eval_config = config.get('evaluation', {})
        
        self.execution_timeout = self.eval_config.get('execution_timeout', 30.0)
        self.num_cpus = self.eval_config.get('num_cpus', 1)
        self.enable_execution_evaluation = self.eval_config.get('enable_execution_evaluation', True)
        
        self.ground_truth_path = self.eval_config.get('ground_truth_sql_path')
        self.difficulty_json_path = self.eval_config.get('difficulty_json_path')
        
        self.ground_truth_sqls = self._load_ground_truth()
        self.difficulty_data = self._load_difficulty_data()
    
    def _load_ground_truth(self) -> Dict[int, str]:

        dev_json_path = self.config.get('data.dev_json_path')
        if dev_json_path and Path(dev_json_path).exists():
            try:
                with open(dev_json_path, 'r') as f:
                    questions = json.load(f)
                
                ground_truth = {}
                for question in questions:
                    question_id = question.get('question_id')
                    sql = question.get('SQL', '')
                    if question_id is not None and sql:
                        ground_truth[question_id] = sql
                
                return ground_truth
            except Exception as e:
                print(f"Error loading ground truth from dev.json: {str(e)}")
        
        if self.ground_truth_path and Path(self.ground_truth_path).exists():
            ground_truth = {}
            try:
                with open(self.ground_truth_path, 'r') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    sql = parts[0]
                                    db_id = parts[1]
                                    ground_truth[line_num] = sql
                            except Exception as e:
                                continue
            except Exception as e:
                print(f"Error loading ground truth: {str(e)}")
            
            return ground_truth
        
        return {}
    
    def _load_difficulty_data(self) -> List[Dict[str, Any]]:

        dev_json_path = self.config.get('data.dev_json_path')
        if dev_json_path and Path(dev_json_path).exists():
            try:
                with open(dev_json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading difficulty data from dev.json: {str(e)}")
        
        if self.difficulty_json_path and Path(self.difficulty_json_path).exists():
            try:
                with open(self.difficulty_json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading difficulty data: {str(e)}")
        
        return []
    
    def evaluate_results(self, pipeline_results: List[Dict[str, Any]]) -> Dict[str, Any]:

        if not pipeline_results:
            return {
                'overall_accuracy': 0.0,
                'execution_results': [],
                'error': 'No results to evaluate'
            }
        
        evaluation_data = self._prepare_evaluation_data(pipeline_results)
        
        exec_results = []
        if self.enable_execution_evaluation:
            exec_results = self._run_execution_evaluation(evaluation_data)
        
        metrics = self._calculate_metrics(exec_results, evaluation_data)
        
        evaluation_result = {
            'overall_accuracy': metrics['overall_accuracy'],
            'execution_success_rate': metrics['execution_success_rate'],
            'difficulty_breakdown': metrics['difficulty_breakdown'],
            'database_breakdown': metrics['database_breakdown'],
            'execution_results': exec_results,
            'evaluation_summary': {
                'total_questions': len(pipeline_results),
                'evaluated_questions': len(exec_results),
                'correct_executions': metrics['correct_count'],
                'successful_executions': metrics['successful_count'],
                'failed_executions': metrics['failed_count']
            }
        }
        
        return evaluation_result
    
    def evaluate_single_question(self, question_result: Dict[str, Any]) -> Dict[str, Any]:

        question_id = question_result['question_id']
        predicted_sql = question_result.get('final_sql', {}).get('selected_sql', '')
        
        result = {
            'question_id': question_id,
            'has_predicted_sql': bool(predicted_sql.strip()),
            'execution_success': False,
            'execution_error': None,
            'is_correct': False,
            'ground_truth_available': question_id in self.ground_truth_sqls
        }
        
        if not predicted_sql.strip():
            result['execution_error'] = 'No SQL generated'
            return result
        
        db_id = question_result.get('db_id')
        if not db_id:
            result['execution_error'] = 'No database ID available'
            return result
            
        db_path = Path(self.config.get('data.databases_path')) / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            result['execution_error'] = f'Database not found: {db_path}'
            return result
        
        try:
            predicted_result = self._execute_sql(str(db_path), predicted_sql, self.execution_timeout)
            if predicted_result is not None:
                result['execution_success'] = True
                result['predicted_execution_result'] = predicted_result
                result['predicted_result_count'] = len(predicted_result)
                
                if result['ground_truth_available']:
                    ground_truth_sql = self.ground_truth_sqls[question_id]
                    try:
                        ground_truth_result = self._execute_sql(str(db_path), ground_truth_sql, self.execution_timeout)
                        if ground_truth_result is not None:
                            result['is_correct'] = self._compare_results(predicted_result, ground_truth_result)
                            result['ground_truth_execution_result'] = ground_truth_result
                            result['ground_truth_result_count'] = len(ground_truth_result)
                            result['ground_truth_sql'] = ground_truth_sql
                    except Exception as e:
                        result['ground_truth_execution_error'] = str(e)
            else:
                result['execution_error'] = 'SQL execution failed'
                
        except Exception as e:
            result['execution_error'] = str(e)
        
        return result
    
    def _execute_sql(self, db_path: str, sql: str, timeout: float = 30.0) -> Optional[List]:

        if not Path(db_path).exists():
            raise Exception(f'Database file not found: {db_path}')
        
        try:
            def execute_with_timeout():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(sql)
                result = cursor.fetchall()
                conn.close()
                return result
            
            try:
                from func_timeout import func_timeout
                return func_timeout(timeout, execute_with_timeout)
            except (ImportError, Exception):
                return execute_with_timeout()
                
        except Exception as e:
            raise Exception(f'SQL execution failed: {str(e)}')
    
    def _compare_results(self, result1: List, result2: List) -> bool:

        if result1 is None or result2 is None:
            return result1 == result2
        
        try:
            set1 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in result1)
            set2 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in result2)
            return set1 == set2
        except (TypeError, ValueError):
            return result1 == result2
    
    def _prepare_evaluation_data(self, pipeline_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        evaluation_data = []
        
        for result in pipeline_results:
            question_id = result['question_id']
            db_id = result['db_id']
            predicted_sql = result.get('final_sql', {}).get('selected_sql', '')
            
            ground_truth_sql = self.ground_truth_sqls.get(question_id, '')
            
            difficulty = 'unknown'
            for question_data in self.difficulty_data:
                if question_data.get('question_id') == question_id:
                    difficulty = question_data.get('difficulty', 'unknown')
                    break
            
            databases_path = Path(self.config.get('data.databases_path', 'data/bird_benchmark/dev_databases'))
            db_path = databases_path / db_id / f"{db_id}.sqlite"
            
            evaluation_data.append({
                'question_id': question_id,
                'db_id': db_id,
                'predicted_sql': predicted_sql,
                'ground_truth_sql': ground_truth_sql,
                'difficulty': difficulty,
                'db_path': str(db_path),
                'original_result': result
            })
        
        return evaluation_data
    
    def _run_execution_evaluation(self, evaluation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        exec_results = []
        
        if self.num_cpus > 1:
            exec_results = self._run_parallel_evaluation(evaluation_data)
        else:
            for i, data in enumerate(evaluation_data):
                result = self._execute_single_comparison(
                    data['predicted_sql'],
                    data['ground_truth_sql'], 
                    data['db_path'],
                    i
                )
                result['question_id'] = data['question_id']
                result['db_id'] = data['db_id']
                result['difficulty'] = data['difficulty']
                exec_results.append(result)
        
        return exec_results
    
    def _run_parallel_evaluation(self, evaluation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        exec_results = []
        
        def result_callback(result):
            exec_results.append(result)
        
        pool = mp.Pool(processes=self.num_cpus)
        
        for i, data in enumerate(evaluation_data):
            pool.apply_async(
                self._execute_single_comparison,
                args=(data['predicted_sql'], data['ground_truth_sql'], data['db_path'], i),
                callback=result_callback
            )
        
        pool.close()
        pool.join()
        
        exec_results.sort(key=lambda x: x['sql_idx'])
        
        for i, result in enumerate(exec_results):
            if i < len(evaluation_data):
                result['question_id'] = evaluation_data[i]['question_id']
                result['db_id'] = evaluation_data[i]['db_id']
                result['difficulty'] = evaluation_data[i]['difficulty']
        
        return exec_results
    
    def _execute_single_comparison(self, predicted_sql: str, ground_truth_sql: str, 
                                  db_path: str, idx: int) -> Dict[str, Any]:

        try:
            result = func_timeout(
                self.execution_timeout,
                self._execute_sql_comparison,
                args=(predicted_sql, ground_truth_sql, db_path)
            )
            return {
                'sql_idx': idx,
                'execution_success': result['execution_success'],
                'correct_result': result['correct_result'],
                'error_message': result.get('error_message', ''),
                'predicted_result_count': result.get('predicted_result_count', 0),
                'ground_truth_result_count': result.get('ground_truth_result_count', 0)
            }
        except FunctionTimedOut:
            return {
                'sql_idx': idx,
                'execution_success': False,
                'correct_result': False,
                'error_message': 'timeout',
                'predicted_result_count': 0,
                'ground_truth_result_count': 0
            }
        except Exception as e:
            return {
                'sql_idx': idx,
                'execution_success': False,
                'correct_result': False,
                'error_message': str(e),
                'predicted_result_count': 0,
                'ground_truth_result_count': 0
            }
    
    def _execute_sql_comparison(self, predicted_sql: str, ground_truth_sql: str, 
                               db_path: str) -> Dict[str, Any]:

        if not Path(db_path).exists():
            return {
                'execution_success': False,
                'correct_result': False,
                'error_message': f'Database file not found: {db_path}'
            }
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            predicted_result = None
            predicted_error = None
            try:
                cursor.execute(predicted_sql)
                predicted_result = cursor.fetchall()
            except Exception as e:
                predicted_error = str(e)
            
            ground_truth_result = None
            ground_truth_error = None
            try:
                cursor.execute(ground_truth_sql)
                ground_truth_result = cursor.fetchall()
            except Exception as e:
                ground_truth_error = str(e)
            
            conn.close()
            
            if predicted_error:
                return {
                    'execution_success': False,
                    'correct_result': False,
                    'error_message': f'Predicted SQL error: {predicted_error}',
                    'predicted_result_count': 0,
                    'ground_truth_result_count': len(ground_truth_result) if ground_truth_result else 0
                }
            
            if ground_truth_error:
                return {
                    'execution_success': False,
                    'correct_result': False,
                    'error_message': f'Ground truth SQL error: {ground_truth_error}',
                    'predicted_result_count': len(predicted_result) if predicted_result else 0,
                    'ground_truth_result_count': 0
                }
            
            execution_success = True
            correct_result = set(predicted_result) == set(ground_truth_result)
            
            return {
                'execution_success': execution_success,
                'correct_result': correct_result,
                'error_message': '',
                'predicted_result_count': len(predicted_result),
                'ground_truth_result_count': len(ground_truth_result)
            }
            
        except Exception as e:
            return {
                'execution_success': False,
                'correct_result': False,
                'error_message': str(e),
                'predicted_result_count': 0,
                'ground_truth_result_count': 0
            }
    
    def _calculate_metrics(self, exec_results: List[Dict[str, Any]], 
                          evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:

        if not exec_results:
            return {
                'overall_accuracy': 0.0,
                'execution_success_rate': 0.0,
                'difficulty_breakdown': {},
                'database_breakdown': {},
                'correct_count': 0,
                'successful_count': 0,
                'failed_count': 0
            }
        
        correct_count = sum(1 for r in exec_results if r['correct_result'])
        successful_count = sum(1 for r in exec_results if r['execution_success'])
        total_count = len(exec_results)
        
        overall_accuracy = correct_count / total_count * 100 if total_count > 0 else 0.0
        execution_success_rate = successful_count / total_count * 100 if total_count > 0 else 0.0
        
        difficulty_stats = {}
        difficulty_groups = {}
        
        for result in exec_results:
            difficulty = result.get('difficulty', 'unknown')
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(result)
        
        for difficulty, results in difficulty_groups.items():
            correct = sum(1 for r in results if r['correct_result'])
            total = len(results)
            accuracy = correct / total * 100 if total > 0 else 0.0
            
            difficulty_stats[difficulty] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        
        database_stats = {}
        database_groups = {}
        
        for result in exec_results:
            db_id = result.get('db_id', 'unknown')
            if db_id not in database_groups:
                database_groups[db_id] = []
            database_groups[db_id].append(result)
        
        for db_id, results in database_groups.items():
            correct = sum(1 for r in results if r['correct_result'])
            total = len(results)
            accuracy = correct / total * 100 if total > 0 else 0.0
            
            database_stats[db_id] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        
        return {
            'overall_accuracy': overall_accuracy,
            'execution_success_rate': execution_success_rate,
            'difficulty_breakdown': difficulty_stats,
            'database_breakdown': database_stats,
            'correct_count': correct_count,
            'successful_count': successful_count,
            'failed_count': total_count - successful_count
        }
    
    def export_evaluation_report(self, evaluation_result: Dict[str, Any], output_path: str) -> None:

        report = {
            'bird_evaluation_report': {
                'overall_metrics': {
                    'accuracy': evaluation_result['overall_accuracy'],
                    'execution_success_rate': evaluation_result['execution_success_rate'],
                    'total_questions': evaluation_result['evaluation_summary']['total_questions'],
                    'correct_executions': evaluation_result['evaluation_summary']['correct_executions']
                },
                'difficulty_breakdown': evaluation_result['difficulty_breakdown'],
                'database_breakdown': evaluation_result['database_breakdown'],
                'detailed_results': evaluation_result['execution_results']
            },
            'export_timestamp': Path(output_path).stem
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_evaluation_summary(self, evaluation_result: Dict[str, Any]) -> None:

        print("\n" + "="*80)
        print("BIRD BENCHMARK EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Execution Accuracy: {evaluation_result['overall_accuracy']:.2f}%")
        print(f"  Execution Success Rate: {evaluation_result['execution_success_rate']:.2f}%")
        print(f"  Total Questions: {evaluation_result['evaluation_summary']['total_questions']}")
        print(f"  Correct Executions: {evaluation_result['evaluation_summary']['correct_executions']}")
        
        print(f"\nDIFFICULTY BREAKDOWN:")
        for difficulty, stats in evaluation_result['difficulty_breakdown'].items():
            print(f"  {difficulty.capitalize()}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2f}%)")
        
        print(f"\nTOP DATABASE PERFORMANCE:")
        db_stats = evaluation_result['database_breakdown']
        sorted_dbs = sorted(db_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for db_id, stats in sorted_dbs[:10]:
            print(f"  {db_id}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2f}%)")
        
        print("\n" + "="*80)
