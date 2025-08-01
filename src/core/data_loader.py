import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from tools.schema_representations.mschema import SchemaHandler

class BirdDataLoader:
    
    def __init__(self, config_manager):

        self.config = config_manager
        self.data_paths = config_manager.get_data_paths()
        
        self.schema_handler = SchemaHandler()
        
        self._questions_cache = None
        self._databases_cache = None
    
    def load_questions(self, force_reload: bool = False) -> List[Dict[str, Any]]:

        if self._questions_cache is not None and not force_reload:
            return self._questions_cache
        
        dev_json_path = Path(self.data_paths['dev_json'])
        if not dev_json_path.exists():
            raise FileNotFoundError(f"BIRD dev.json not found at {dev_json_path}")
        
        with open(dev_json_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        required_fields = ['question_id', 'db_id', 'question', 'SQL', 'difficulty']
        for i, question in enumerate(questions):
            for field in required_fields:
                if field not in question:
                    raise ValueError(f"Missing field '{field}' in question {i}")
        
        self._questions_cache = questions
        return questions
    
    def load_databases(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:

        if self._databases_cache is not None and not force_reload:
            return self._databases_cache
        
        databases_path = Path(self.data_paths['databases'])
        if not databases_path.exists():
            raise FileNotFoundError(f"BIRD databases directory not found at {databases_path}")
        
        databases = {}
        
        for db_dir in databases_path.iterdir():
            if db_dir.is_dir():
                db_id = db_dir.name
                sqlite_file = db_dir / f"{db_id}.sqlite"
                
                if sqlite_file.exists():
                    try:
                        db_info = self.schema_handler.get_database_info(str(sqlite_file), db_id)
                        
                        db_info['database_path'] = str(sqlite_file)
                        db_info['description_files'] = self._load_description_files(db_dir)
                        db_info['table_descriptions'] = self._load_table_descriptions(db_dir)
                        
                        databases[db_id] = db_info
                        
                    except Exception as e:
                        print(f"Warning: Failed to load database {db_id}: {str(e)}")
                        continue
        
        if not databases:
            raise ValueError("No valid databases found in the BIRD dataset")
        
        self._databases_cache = databases
        return databases
    
    def _load_description_files(self, db_dir: Path) -> Dict[str, str]:

        description_files = {}
        desc_dir = db_dir / "database_description"
        
        if desc_dir.exists():
            for csv_file in desc_dir.glob("*.csv"):
                table_name = csv_file.stem
                try:
                    content = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(csv_file, 'r', encoding=encoding) as f:
                                content = f.read()
                                break
                        except UnicodeDecodeError:
                            continue
                    
                    if content:
                        description_files[table_name] = content
                    else:
                        print(f"Warning: Failed to read description file {csv_file}: encoding issues")
                except Exception as e:
                    print(f"Warning: Failed to read description file {csv_file}: {str(e)}")
        
        return description_files
    
    def _load_table_descriptions(self, db_dir: Path) -> Dict[str, Dict[str, Any]]:

        table_descriptions = {}
        desc_dir = db_dir / "database_description"
        
        if desc_dir.exists():
            for csv_file in desc_dir.glob("*.csv"):
                table_name = csv_file.stem
                try:
                    df = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(csv_file, encoding=encoding)
                            break
                        except (UnicodeDecodeError, pd.errors.ParserError):
                            continue
                    
                    if df is not None:
                        if 'column_name' in df.columns:
                            columns_info = {}
                            for _, row in df.iterrows():
                                col_name = row.get('column_name', row.get('original_column_name', ''))
                                if col_name:
                                    columns_info[col_name] = {
                                        'description': row.get('column_description', ''),
                                        'data_format': row.get('data_format', ''),
                                        'value_description': row.get('value_description', '')
                                    }
                            
                            table_descriptions[table_name] = {
                                'type': 'structured',
                                'columns': columns_info
                            }
                        else:
                            content = None
                            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                                try:
                                    with open(csv_file, 'r', encoding=encoding) as f:
                                        content = f.read()
                                        break
                                except UnicodeDecodeError:
                                    continue
                            
                            if content:
                                table_descriptions[table_name] = {
                                    'type': 'raw',
                                    'content': content
                                }
                    else:
                        content = None
                        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                with open(csv_file, 'r', encoding=encoding) as f:
                                    content = f.read()
                                    break
                            except UnicodeDecodeError:
                                continue
                        
                        if content:
                            table_descriptions[table_name] = {
                                'type': 'raw',
                                'content': content
                            }
                        
                except Exception as e:
                    content = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(csv_file, 'r', encoding=encoding) as f:
                                content = f.read()
                                break
                        except UnicodeDecodeError:
                            continue
                    
                    if content:
                        table_descriptions[table_name] = {
                            'type': 'raw',
                            'content': content
                        }
                    else:
                        print(f"Warning: Failed to read table description {csv_file}: {str(e)}")
        
        return table_descriptions
    
    def get_question_by_id(self, question_id: int) -> Optional[Dict[str, Any]]:

        questions = self.load_questions()
        for question in questions:
            if question['question_id'] == question_id:
                return question
        return None
    
    def get_questions_by_database(self, db_id: str) -> List[Dict[str, Any]]:

        questions = self.load_questions()
        return [q for q in questions if q['db_id'] == db_id]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:

        questions = self.load_questions()
        return [q for q in questions if q['difficulty'] == difficulty]
    
    def get_database_info(self, db_id: str) -> Optional[Dict[str, Any]]:

        databases = self.load_databases()
        return databases.get(db_id)
    
    def get_dataset_statistics(self) -> Dict[str, Any]:

        questions = self.load_questions()
        databases = self.load_databases()
        
        difficulty_counts = {}
        db_counts = {}
        
        for question in questions:
            difficulty = question['difficulty']
            db_id = question['db_id']
            
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            db_counts[db_id] = db_counts.get(db_id, 0) + 1
        
        total_tables = 0
        total_columns = 0
        
        for db_info in databases.values():
            schema_engine = db_info.get('schema_engine')
            if schema_engine:
                total_tables += len(schema_engine._usable_tables)
                for table_name in schema_engine._usable_tables:
                    try:
                        columns = schema_engine.inspector.get_columns(table_name)
                        total_columns += len(columns)
                    except Exception:
                        continue
        
        return {
            'total_questions': len(questions),
            'total_databases': len(databases),
            'total_tables': total_tables,
            'total_columns': total_columns,
            'difficulty_distribution': difficulty_counts,
            'database_distribution': db_counts,
            'available_databases': list(databases.keys())
        }
    
    def validate_dataset(self) -> Dict[str, Any]:

        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            questions = self.load_questions()
            if not questions:
                validation_results['errors'].append("No questions found in dev.json")
                validation_results['valid'] = False
            
            databases = self.load_databases()
            referenced_dbs = set(q['db_id'] for q in questions)
            available_dbs = set(databases.keys())
            
            missing_dbs = referenced_dbs - available_dbs
            if missing_dbs:
                validation_results['errors'].append(f"Missing databases: {missing_dbs}")
                validation_results['valid'] = False
            
            unused_dbs = available_dbs - referenced_dbs
            if unused_dbs:
                validation_results['warnings'].append(f"Unused databases: {unused_dbs}")
            
            for i, question in enumerate(questions):
                if not question.get('question', '').strip():
                    validation_results['errors'].append(f"Empty question text in question {i}")
                if not question.get('SQL', '').strip():
                    validation_results['errors'].append(f"Empty SQL in question {i}")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['valid'] = False
        
        return validation_results
    
    def export_statistics(self, output_path: str) -> None:

        stats = self.get_dataset_statistics()
        validation = self.validate_dataset()
        
        export_data = {
            'statistics': stats,
            'validation': validation,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
