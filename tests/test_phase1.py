import sys
import os
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config_manager import ConfigManager
from src.core.model_manager import ModelManager
from src.core.data_loader import BirdDataLoader
from src.phases.phase1_schema_linking import SchemaLinker


def test_schema_linking():

    config = ConfigManager()
    data_loader = BirdDataLoader(config)
    
    assert config is not None
    assert data_loader is not None
    
    questions = data_loader.load_questions()
    databases = data_loader.load_databases()
    
    assert len(questions) > 0, "No questions loaded"
    assert len(databases) > 0, "No databases loaded"
    
    test_question = questions[0]
    test_db_id = test_question['db_id']
    
    assert test_db_id in databases, f"Database {test_db_id} not found"
    assert 'question' in test_question, "Question text missing"
    assert 'SQL' in test_question, "SQL missing"
    
    db_info = databases[test_db_id]
    assert 'mschema' in db_info, "M-Schema missing from database info"
    assert 'schema_engine' in db_info, "Schema engine missing"


def test_mschema_generation():

    from tools.schema_representations.mschema import SchemaHandler
    
    config = ConfigManager()
    data_loader = BirdDataLoader(config)
    
    databases = data_loader.load_databases()
    assert len(databases) > 0, "No databases available for M-Schema test"
    
    first_db_id = list(databases.keys())[0]
    db_info = databases[first_db_id]
    
    schema_engine = db_info['schema_engine']
    mschema_str = schema_engine.get_mschema_string(example_num=2)
    
    assert len(mschema_str) > 0, "Empty M-Schema generated"
    assert 'DB_ID' in mschema_str, "M-Schema missing DB_ID"
    assert 'Schema' in mschema_str, "M-Schema missing Schema section"
