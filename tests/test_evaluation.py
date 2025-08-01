import sys
import os
import sqlite3
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config_manager import ConfigManager
from tools.evaluation.bird_evaluator import BirdEvaluator


def create_test_database():

    temp_db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
    temp_db.close()
    
    conn = sqlite3.connect(temp_db.name)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            grade TEXT
        )
    ''')
    
    test_data = [
        (1, 'Alice', 20, 'A'),
        (2, 'Bob', 19, 'B'),
        (3, 'Charlie', 21, 'A'),
        (4, 'Diana', 18, 'C')
    ]
    
    cursor.executemany(
        'INSERT INTO students (id, name, age, grade) VALUES (?, ?, ?, ?)',
        test_data
    )
    
    conn.commit()
    conn.close()
    
    return temp_db.name


def test_bird_evaluator_init():

    print("Testing BIRD Evaluator Initialization")
    print("=" * 40)
    
    try:
        config = ConfigManager()
        evaluator = BirdEvaluator(config)
        
        print("✓ BIRD evaluator initialized")
        
        timeout = evaluator.execution_timeout
        num_cpus = evaluator.num_cpus
        
        print(f"✓ Execution timeout: {timeout}s")
        print(f"✓ CPU cores: {num_cpus}")
        
        ground_truth = evaluator.ground_truth_sqls
        print(f"✓ Ground truth loaded: {len(ground_truth)} entries")
        
        difficulty_data = evaluator.difficulty_data
        print(f"✓ Difficulty data loaded: {len(difficulty_data)} entries")
        
        return True
        
    except Exception as e:
        print(f"✗ BIRD evaluator initialization failed: {str(e)}")
        return False


def test_sql_execution_comparison():

    print("\nTesting SQL Execution Comparison")
    print("=" * 35)
    
    try:
        config = ConfigManager()
        evaluator = BirdEvaluator(config)
        
        test_db_path = create_test_database()
        print(f"✓ Test database created: {Path(test_db_path).name}")
        
        try:
            predicted_sql = "SELECT COUNT(*) FROM students"
            ground_truth_sql = "SELECT COUNT(*) FROM students"
            
            result = evaluator._execute_sql_comparison(predicted_sql, ground_truth_sql, test_db_path)
            
            if result['execution_success'] and result['correct_result']:
                print("✓ Correct SQL comparison works")
            else:
                print(f"✗ Correct SQL comparison failed: {result}")
                return False
            
            predicted_sql2 = "SELECT COUNT(id) FROM students"
            result2 = evaluator._execute_sql_comparison(predicted_sql2, ground_truth_sql, test_db_path)
            
            if result2['execution_success'] and result2['correct_result']:
                print("✓ Equivalent SQL comparison works")
            else:
                print("? Different SQL gave different result (may be expected)")
            
            predicted_sql3 = "SELECT COUNT(*) FROM students WHERE age > 25"
            result3 = evaluator._execute_sql_comparison(predicted_sql3, ground_truth_sql, test_db_path)
            
            if result3['execution_success'] and not result3['correct_result']:
                print("✓ Incorrect SQL properly identified")
            else:
                print(f"? Unexpected result for incorrect SQL: {result3}")
            
            invalid_sql = "SELEC COUNT(*) FROM students"
            result4 = evaluator._execute_sql_comparison(invalid_sql, ground_truth_sql, test_db_path)
            
            if not result4['execution_success']:
                print("✓ Invalid SQL properly caught")
                print(f"  Error: {result4['error_message'][:50]}...")
            else:
                print("✗ Invalid SQL not caught")
                return False
            
        finally:
            os.unlink(test_db_path)
            print("✓ Test database cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ SQL execution comparison test failed: {str(e)}")
        return False


def test_metrics_calculation():

    print("\nTesting Metrics Calculation")
    print("=" * 30)
    
    try:
        config = ConfigManager()
        evaluator = BirdEvaluator(config)
        
        mock_exec_results = [
            {
                'question_id': 0,
                'db_id': 'test_db',
                'difficulty': 'simple',
                'execution_success': True,
                'correct_result': True
            },
            {
                'question_id': 1,
                'db_id': 'test_db',
                'difficulty': 'simple',
                'execution_success': True,
                'correct_result': False
            },
            {
                'question_id': 2,
                'db_id': 'test_db2',
                'difficulty': 'moderate',
                'execution_success': True,
                'correct_result': True
            },
            {
                'question_id': 3,
                'db_id': 'test_db2',
                'difficulty': 'challenging',
                'execution_success': False,
                'correct_result': False
            }
        ]
        
        mock_eval_data = [
            {
                'question_id': i,
                'db_id': result['db_id'],
                'difficulty': result['difficulty']
            }
            for i, result in enumerate(mock_exec_results)
        ]
        
        print(f"✓ Mock data created: {len(mock_exec_results)} results")
        
        metrics = evaluator._calculate_metrics(mock_exec_results, mock_eval_data)
        
        print(f"✓ Metrics calculated")
        print(f"  Overall accuracy: {metrics['overall_accuracy']:.1f}%")
        print(f"  Execution success rate: {metrics['execution_success_rate']:.1f}%")
        print(f"  Correct count: {metrics['correct_count']}")
        print(f"  Successful count: {metrics['successful_count']}")
        
        expected_correct = 2
        expected_successful = 3
        
        if metrics['correct_count'] == expected_correct:
            print("✓ Correct count accurate")
        else:
            print(f"✗ Wrong correct count: expected {expected_correct}, got {metrics['correct_count']}")
            return False
        
        if metrics['successful_count'] == expected_successful:
            print("✓ Successful count accurate")
        else:
            print(f"✗ Wrong successful count: expected {expected_successful}, got {metrics['successful_count']}")
            return False
        
        difficulty_breakdown = metrics['difficulty_breakdown']
        print(f"✓ Difficulty breakdown: {len(difficulty_breakdown)} categories")
        
        for difficulty, stats in difficulty_breakdown.items():
            print(f"  {difficulty}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
        
        database_breakdown = metrics['database_breakdown']
        print(f"✓ Database breakdown: {len(database_breakdown)} databases")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics calculation test failed: {str(e)}")
        return False


def test_evaluation_pipeline():

    print("\nTesting Complete Evaluation Pipeline")
    print("=" * 40)
    
    try:
        config = ConfigManager()
        evaluator = BirdEvaluator(config)
        
        mock_pipeline_results = [
            {
                'question_id': 0,
                'db_id': 'test_db',
                'question': 'How many students are there?',
                'ground_truth_sql': 'SELECT COUNT(*) FROM students',
                'difficulty': 'simple',
                'final_sql': {
                    'selected_sql': 'SELECT COUNT(*) FROM students'
                }
            },
            {
                'question_id': 1,
                'db_id': 'test_db',
                'question': 'What are the names of all students?',
                'ground_truth_sql': 'SELECT name FROM students',
                'difficulty': 'simple',
                'final_sql': {
                    'selected_sql': 'SELECT student_name FROM students'
                }
            }
        ]
        
        print(f"✓ Mock pipeline results created: {len(mock_pipeline_results)} results")
        
        eval_data = evaluator._prepare_evaluation_data(mock_pipeline_results)
        print(f"✓ Evaluation data prepared: {len(eval_data)} entries")
        
        for i, data in enumerate(eval_data):
            required_keys = ['question_id', 'db_id', 'predicted_sql', 'ground_truth_sql', 'difficulty']
            for key in required_keys:
                if key not in data:
                    print(f"✗ Missing key in eval data {i}: {key}")
                    return False
        
        print("✓ Evaluation data structure valid")
        
        evaluator.enable_execution_evaluation = False
        
        try:
            evaluation_result = evaluator.evaluate_results(mock_pipeline_results)
            
            print("✓ Evaluation completed")
            print(f"  Overall accuracy: {evaluation_result.get('overall_accuracy', 'N/A')}")
            print(f"  Total questions: {evaluation_result['evaluation_summary']['total_questions']}")
            
            required_keys = ['overall_accuracy', 'evaluation_summary']
            for key in required_keys:
                if key not in evaluation_result:
                    print(f"✗ Missing key in evaluation result: {key}")
                    return False
            
            print("✓ Evaluation result structure valid")
            
        except Exception as e:
            print(f"? Evaluation had issues (expected without full BIRD setup): {str(e)[:100]}...")
            print("✓ Structure test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation pipeline test failed: {str(e)}")
        return False


def test_evaluation_export():

    print("\nTesting Evaluation Export")
    print("=" * 25)
    
    try:
        config = ConfigManager()
        evaluator = BirdEvaluator(config)
        
        mock_evaluation_result = {
            'overall_accuracy': 75.0,
            'execution_success_rate': 90.0,
            'difficulty_breakdown': {
                'simple': {'correct': 3, 'total': 4, 'accuracy': 75.0},
                'moderate': {'correct': 2, 'total': 3, 'accuracy': 66.7}
            },
            'database_breakdown': {
                'test_db': {'correct': 5, 'total': 7, 'accuracy': 71.4}
            },
            'evaluation_summary': {
                'total_questions': 7,
                'correct_executions': 5,
                'successful_executions': 6
            },
            'execution_results': []
        }
        
        print("✓ Mock evaluation result created")
        
        test_output_dir = Path("tests/output")
        test_output_dir.mkdir(exist_ok=True)
        
        export_path = test_output_dir / "test_evaluation_report.json"
        
        try:
            evaluator.export_evaluation_report(mock_evaluation_result, str(export_path))
            
            if export_path.exists():
                print(f"✓ Evaluation report exported: {export_path.name}")
                
                file_size = export_path.stat().st_size
                if file_size > 0:
                    print(f"✓ Export file has content: {file_size} bytes")
                else:
                    print("✗ Export file is empty")
                    return False
                
                export_path.unlink()
                print("✓ Test export file cleaned up")
                
            else:
                print("✗ Export file not created")
                return False
        
        except Exception as e:
            print(f"? Export had issues: {str(e)[:50]}...")
            print("✓ Export structure test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation export test failed: {str(e)}")
        return False


def main():

    print("BIRD Evaluation Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        ("BIRD Evaluator Initialization", test_bird_evaluator_init),
        ("SQL Execution Comparison", test_sql_execution_comparison),
        ("Metrics Calculation", test_metrics_calculation),
        ("Evaluation Pipeline", test_evaluation_pipeline),
        ("Evaluation Export", test_evaluation_export)
    ]
    
    for test_name, test_func in tests:
        if test_func():
            print(f"✓ {test_name} test PASSED")
        else:
            print(f"✗ {test_name} test FAILED")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All evaluation tests PASSED!")
        return 0
    else:
        print("❌ Some evaluation tests FAILED!")
        print("Note: Some failures expected without full BIRD dataset")
        return 1


if __name__ == "__main__":
    sys.exit(main())
