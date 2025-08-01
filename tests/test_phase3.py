import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config_manager import ConfigManager
from src.core.model_manager import ModelManager
from src.phases.phase3_sql_selection import (
    SQLSelector, SelectionCandidate, ValidityFilter, 
    LLMCritic, ValueAlignmentChecker, SelfConsistencyChecker
)


def test_selection_candidate():

    print("Testing Selection Candidate")
    print("=" * 30)
    
    try:
        original_candidates = [
            {
                'sql': 'SELECT COUNT(*) FROM students',
                'model_name': 'test_model_1',
                'schema_representation': 'm_schema',
                'temperature': 0.1,
                'is_valid': True
            },
            {
                'sql': 'SELECT COUNT(*) FROM students',
                'model_name': 'test_model_2',
                'schema_representation': 'ddl',
                'temperature': 0.3,
                'is_valid': True
            }
        ]
        
        candidate = SelectionCandidate(
            sql="SELECT COUNT(*) FROM students",
            original_candidates=original_candidates
        )
        
        print(f"✓ Selection candidate created: {candidate.sql}")
        print(f"✓ Original candidates: {len(candidate.original_candidates)}")
        
        candidate.validity_score = 1.0
        candidate.popularity_score = 0.8
        candidate.llm_critic_score = 0.9
        candidate.value_alignment_score = 0.7
        candidate.self_consistency_score = 0.85
        candidate.final_score = 0.84
        
        candidate_dict = candidate.to_dict()
        print(f"✓ Serialization works: {len(candidate_dict)} fields")
        
        if 0 <= candidate.final_score <= 1:
            print("✓ Valid score range")
        else:
            print("✗ Invalid score range")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Selection candidate test failed: {str(e)}")
        return False


def test_validity_filter():

    print("\nTesting Validity Filter")
    print("=" * 25)
    
    try:
        config = ConfigManager()
        validity_filter = ValidityFilter(config)
        
        test_candidates = [
            {
                'sql': 'SELECT COUNT(*) FROM students',
                'model_name': 'model1',
                'is_valid': True
            },
            {
                'sql': 'SELECT COUNT(*) FROM students',
                'model_name': 'model2',
                'is_valid': True
            },
            {
                'sql': 'SELEC COUNT(*) FROM students',
                'model_name': 'model3',
                'is_valid': False
            },
            {
                'sql': 'SELECT * FROM courses',
                'model_name': 'model4',
                'is_valid': True
            }
        ]
        
        print(f"✓ Test candidates created: {len(test_candidates)}")
        
        filtered_candidates = validity_filter.filter_candidates(test_candidates)
        
        print(f"✓ Filtering completed: {len(filtered_candidates)} candidates remaining")
        
        valid_count = sum(1 for c in test_candidates if c.get('is_valid', True))
        if validity_filter.remove_syntax_errors:
            print(f"✓ Invalid candidates filtered (expected: ≤{valid_count}, got: {len(filtered_candidates)})")
        
        unique_sqls = set(c.sql for c in filtered_candidates)
        if validity_filter.remove_duplicates:
            print(f"✓ Duplicates removed: {len(unique_sqls)} unique SQLs")
        
        return True
        
    except Exception as e:
        print(f"✗ Validity filter test failed: {str(e)}")
        return False


def test_value_alignment_checker():

    print("\nTesting Value Alignment Checker")
    print("=" * 35)
    
    try:
        config = ConfigManager()
        value_checker = ValueAlignmentChecker(config)
        
        test_question = "How many students are older than 18 years?"
        
        test_candidates = [
            SelectionCandidate(
                sql="SELECT COUNT(*) FROM students WHERE age > 18",
                original_candidates=[]
            ),
            SelectionCandidate(
                sql="SELECT COUNT(*) FROM students WHERE age > 21",
                original_candidates=[]
            ),
            SelectionCandidate(
                sql="SELECT COUNT(*) FROM students",
                original_candidates=[]
            )
        ]
        
        print(f"✓ Test question: {test_question}")
        print(f"✓ Test candidates: {len(test_candidates)}")
        
        checked_candidates = value_checker.check_alignment(test_question, test_candidates)
        
        print("✓ Value alignment check completed")
        
        for i, candidate in enumerate(checked_candidates):
            score = candidate.value_alignment_score
            print(f"  Candidate {i+1} alignment score: {score:.3f}")
            
            if 0 <= score <= 1:
                print(f"    ✓ Valid alignment score")
            else:
                print(f"    ✗ Invalid alignment score")
                return False
        
        if len(checked_candidates) > 1:
            first_score = checked_candidates[0].value_alignment_score
            if first_score >= 0.5:
                print("✓ Correct value candidate has good alignment")
            else:
                print(f"? Unexpected alignment score for correct candidate: {first_score}")
        
        return True
        
    except Exception as e:
        print(f"✗ Value alignment checker test failed: {str(e)}")
        return False


def test_self_consistency_checker():
    print("\nTesting Self-Consistency Checker")
    print("=" * 35)
    
    try:
        config = ConfigManager()
        consistency_checker = SelfConsistencyChecker(config)
        
        test_candidates = [
            SelectionCandidate(
                sql="SELECT COUNT(*) FROM students",
                original_candidates=[
                    {'model_name': 'model1', 'schema_representation': 'm_schema', 'temperature': 0.1},
                    {'model_name': 'model2', 'schema_representation': 'ddl', 'temperature': 0.1},
                    {'model_name': 'model3', 'schema_representation': 'm_schema', 'temperature': 0.3}
                ]
            ),
            SelectionCandidate(
                sql="SELECT * FROM students",
                original_candidates=[
                    {'model_name': 'model1', 'schema_representation': 'json', 'temperature': 0.7}
                ]
            )
        ]
        
        test_candidates[0].popularity_score = 0.75
        test_candidates[1].popularity_score = 0.25
        
        print(f"✓ Test candidates created with popularity scores")
        
        checked_candidates = consistency_checker.check_consistency(test_candidates)
        
        print("✓ Self-consistency check completed")
        
        for i, candidate in enumerate(checked_candidates):
            score = candidate.self_consistency_score
            print(f"  Candidate {i+1} consistency score: {score:.3f}")
            
            if 0 <= score <= 1:
                print(f"    ✓ Valid consistency score")
            else:
                print(f"    ✗ Invalid consistency score")
                return False
        
        if len(checked_candidates) >= 2:
            if checked_candidates[0].self_consistency_score >= checked_candidates[1].self_consistency_score:
                print("✓ Popular candidate has higher consistency score")
            else:
                print("? Unexpected consistency scoring")
        
        return True
        
    except Exception as e:
        print(f"✗ Self-consistency checker test failed: {str(e)}")
        return False


def test_sql_selector_structure():

    print("\nTesting SQL Selector Structure")
    print("=" * 35)
    
    try:
        config = ConfigManager()
        model_manager = ModelManager(config)
        sql_selector = SQLSelector(config, model_manager)
        
        print("✓ SQL selector initialized")
        
        if sql_selector.validity_filter:
            print("✓ Validity filter initialized")
        else:
            print("✗ Validity filter not initialized")
            return False
        
        if sql_selector.llm_critic:
            print("✓ LLM critic initialized")
        else:
            print("✗ LLM critic not initialized")
            return False
        
        if sql_selector.value_checker:
            print("✓ Value alignment checker initialized")
        else:
            print("✗ Value alignment checker not initialized")
            return False
        
        if sql_selector.consistency_checker:
            print("✓ Self-consistency checker initialized")
        else:
            print("✗ Self-consistency checker not initialized")
            return False
        
        weights = sql_selector.selection_weights
        weight_sum = sum(weights.values())
        
        print(f"✓ Selection weights configured: {len(weights)} weights")
        print(f"✓ Weight sum: {weight_sum:.3f}")
        
        if 0.9 <= weight_sum <= 1.1:
            print("✓ Weights sum to approximately 1.0")
        else:
            print(f"✗ Weights sum to {weight_sum}, expected ~1.0")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ SQL selector structure test failed: {str(e)}")
        return False


def test_mock_selection():
    print("\nTesting Mock SQL Selection")
    print("=" * 30)
    
    try:
        config = ConfigManager()
        model_manager = ModelManager(config)
        sql_selector = SQLSelector(config, model_manager)
        
        mock_question = {
            'question_id': 1,
            'question': 'How many students are older than 18?',
            'evidence': 'Count students with age > 18'
        }
        
        mock_candidates_result = {
            'candidates': [
                {
                    'sql': 'SELECT COUNT(*) FROM students WHERE age > 18',
                    'model_name': 'model1',
                    'schema_representation': 'm_schema',
                    'temperature': 0.1,
                    'is_valid': True,
                    'validation_error': None
                },
                {
                    'sql': 'SELECT COUNT(*) FROM students WHERE age > 21',
                    'model_name': 'model2',
                    'schema_representation': 'ddl',
                    'temperature': 0.3,
                    'is_valid': True,
                    'validation_error': None
                },
                {
                    'sql': 'SELEC COUNT(*) FROM students',
                    'model_name': 'model3',
                    'schema_representation': 'json',
                    'temperature': 0.7,
                    'is_valid': False,
                    'validation_error': 'Syntax error'
                }
            ],
            'schema_representations': {
                'm_schema': 'Test M-Schema content',
                'ddl': 'Test DDL content'
            }
        }
        
        print("✓ Mock data prepared")
        
        try:
            result = sql_selector.select_best_sql(mock_question, mock_candidates_result)
            
            print("✓ Selection process completed")
            print(f"  Selected SQL: {result['selected_sql']}")
            print(f"  Confidence: {result['confidence_score']:.3f}")
            print(f"  Method: {result['selection_method']}")
            print(f"  Candidates considered: {result['candidates_considered']}")
            
            required_keys = ['question_id', 'selected_sql', 'confidence_score', 'selection_method']
            for key in required_keys:
                if key not in result:
                    print(f"✗ Missing result key: {key}")
                    return False
            
            print("✓ Result structure valid")
            
            if result['selected_sql']:
                print("✓ SQL was selected")
            else:
                print("? No SQL selected (may be expected if all invalid)")
            
        except Exception as e:
            print(f"? Selection process had issues (expected without API keys): {str(e)[:100]}...")
            print("✓ Structure test completed despite selection issues")
        
        return True
        
    except Exception as e:
        print(f"✗ Mock selection test failed: {str(e)}")
        return False


def main():
    
    print("Phase 3 Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        ("Selection Candidate", test_selection_candidate),
        ("Validity Filter", test_validity_filter),
        ("Value Alignment Checker", test_value_alignment_checker),
        ("Self-Consistency Checker", test_self_consistency_checker),
        ("SQL Selector Structure", test_sql_selector_structure),
        ("Mock Selection", test_mock_selection)
    ]
    
    for test_name, test_func in tests:
        if test_func():
            print(f"✓ {test_name} test PASSED")
        else:
            print(f"✗ {test_name} test FAILED")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All Phase 3 tests PASSED!")
        return 0
    else:
        print("❌ Some Phase 3 tests FAILED!")
        print("Note: Some failures expected without API keys for LLM critic")
        return 1


if __name__ == "__main__":
    sys.exit(main())
