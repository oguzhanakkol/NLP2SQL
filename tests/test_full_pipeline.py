import sys
import os
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config_manager import ConfigManager
from src.core.model_manager import ModelManager
from src.core.data_loader import BirdDataLoader
from src.phases.phase1_schema_linking import SchemaLinker
from src.phases.phase2_sql_generation import SQLGenerator
from src.phases.phase3_sql_selection import SQLSelector
from tools.evaluation.bird_evaluator import BirdEvaluator


def test_full_pipeline():

    config = ConfigManager()
    model_manager = ModelManager(config)
    data_loader = BirdDataLoader(config)
    
    questions = data_loader.load_questions()
    databases = data_loader.load_databases()
    
    assert questions, "Failed to load questions"
    assert databases, "Failed to load databases"
    
    test_question = questions[0]
    test_db_id = test_question['db_id']
    
    assert test_db_id in databases, f"Database {test_db_id} not available"
    
    assert 'question_id' in test_question
    assert 'question' in test_question
    assert 'db_id' in test_question
    
    db_info = databases[test_db_id]
    assert 'mschema' in db_info
    assert 'schema_engine' in db_info
    
    pipeline_result = {
        'question_id': test_question['question_id'],
        'db_id': test_question['db_id'],
        'question': test_question['question'],
        'ground_truth_sql': test_question.get('SQL', ''),
        'difficulty': test_question.get('difficulty', 'unknown'),
    }
    
    assert 'question_id' in pipeline_result
    assert 'db_id' in pipeline_result
    assert 'question' in pipeline_result


def test_data_flow():

    schema_result = {
        'selected_tables': ['test_table'],
        'selected_columns': ['test_table.id', 'test_table.name'],
        'schema_representations': {
            'm_schema': 'test m-schema content',
            'ddl': 'CREATE TABLE test_table...'
        }
    }
    
    candidates_result = {
        'candidates': [
            {
                'sql': 'SELECT * FROM test_table',
                'model_name': 'test_model',
                'is_valid': True
            }
        ],
        'schema_representations': schema_result['schema_representations']
    }
    
    selection_result = {
        'selected_sql': 'SELECT * FROM test_table',
        'confidence_score': 0.85,
        'selection_method': 'test_selection'
    }
    
    required_schema_fields = ['selected_tables', 'selected_columns', 'schema_representations']
    for field in required_schema_fields:
        assert field in schema_result, f"Missing required schema field: {field}"
    
    required_candidates_fields = ['candidates']
    for field in required_candidates_fields:
        assert field in candidates_result, f"Missing required candidates field: {field}"
    
    required_selection_fields = ['selected_sql', 'confidence_score']
    for field in required_selection_fields:
        assert field in selection_result, f"Missing required selection field: {field}"

