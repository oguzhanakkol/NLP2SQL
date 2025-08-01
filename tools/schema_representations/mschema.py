import sqlite3
import datetime
import decimal
import re
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import create_engine, MetaData, Table, Column, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.inspection import inspect

def write_json(path, data):

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def is_email(string):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    match = re.match(pattern, string)
    return bool(match)


def examples_to_str(examples: list) -> List[str]:
    values = examples.copy()
    
    for i in range(len(values)):
        if isinstance(values[i], datetime.date):
            values = [values[i]]
            break
        elif isinstance(values[i], datetime.datetime):
            values = [values[i]]
            break
        elif isinstance(values[i], decimal.Decimal):
            values[i] = str(float(values[i]))
        elif is_email(str(values[i])):
            values = []
            break
        elif 'http://' in str(values[i]) or 'https://' in str(values[i]):
            values = []
            break
        elif values[i] is not None and '.com' in str(values[i]):
            pass

    return [str(v) for v in values if v is not None and len(str(v)) > 0]


class MSchema:
    
    def __init__(self, db_id: str = 'Anonymous', schema: Optional[str] = None):

        self.db_id = db_id
        self.schema = schema
        self.tables = {}
        self.foreign_keys = []

    def add_table(self, name: str, fields: Dict = None, comment: Optional[str] = None):

        if fields is None:
            fields = {}
        self.tables[name] = {
            "fields": fields.copy(), 
            'examples': [], 
            'comment': comment
        }

    def add_field(self, table_name: str, field_name: str, field_type: str = "",
                  primary_key: bool = False, nullable: bool = True, default: Any = None,
                  autoincrement: bool = False, comment: str = "", examples: List = None, **kwargs):

        if examples is None:
            examples = []
            
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default if default is None else f'{default}',
            "autoincrement": autoincrement,
            "comment": comment,
            "examples": examples.copy(),
            **kwargs
        }

    def add_foreign_key(self, table_name: str, field_name: str, ref_schema: str, 
                       ref_table_name: str, ref_field_name: str):

        self.foreign_keys.append([table_name, field_name, ref_schema, ref_table_name, ref_field_name])

    def get_field_type(self, field_type: str, simple_mode: bool = True) -> str:

        if not simple_mode:
            return field_type
        else:
            return field_type.split("(")[0]

    def single_table_mschema(self, table_name: str, selected_columns: List = None,
                            example_num: int = 3, show_type_detail: bool = False) -> str:

        table_info = self.tables.get(table_name, {})
        output = []
        
        table_comment = table_info.get('comment', '')
        if table_comment and table_comment != 'None' and len(table_comment) > 0:
            if self.schema and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}, {table_comment}")
            else:
                output.append(f"# Table: {table_name}, {table_comment}")
        else:
            if self.schema and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}")
            else:
                output.append(f"# Table: {table_name}")

        field_lines = []
        
        for field_name, field_info in table_info['fields'].items():
            if selected_columns is not None and field_name.lower() not in selected_columns:
                continue

            raw_type = self.get_field_type(field_info['type'], not show_type_detail)
            field_line = f"({field_name}:{raw_type.upper()}"
            
            if field_info['comment']:
                field_line += f", {field_info['comment'].strip()}"

            if field_info.get('primary_key', False):
                field_line += f", Primary Key"

            examples = field_info.get('examples', [])
            if examples and example_num > 0:
                examples = [s for s in examples if s is not None]
                examples = examples_to_str(examples)
                
                if len(examples) > example_num:
                    examples = examples[:example_num]

                if raw_type.upper() in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
                    examples = [examples[0]] if examples else []
                elif examples and max([len(str(s)) for s in examples]) > 20:
                    if max([len(str(s)) for s in examples]) > 50:
                        examples = []
                    else:
                        examples = [examples[0]]

                if examples:
                    example_str = ', '.join([str(example) for example in examples])
                    field_line += f", Examples: [{example_str}]"
            
            field_line += ")"
            field_lines.append(field_line)

        output.append('[')
        output.append(',\n'.join(field_lines))
        output.append(']')

        return '\n'.join(output)

    def to_mschema(self, selected_tables: List = None, selected_columns: List = None,
                  example_num: int = 3, show_type_detail: bool = False) -> str:
        output = []

        output.append(f"【DB_ID】 {self.db_id}")
        output.append(f"【Schema】")

        if selected_tables is not None:
            selected_tables = [s.lower() for s in selected_tables]
        if selected_columns is not None:
            selected_columns = [s.lower() for s in selected_columns]
            selected_tables = [s.split('.')[0].lower() for s in selected_columns]

        for table_name, table_info in self.tables.items():
            if selected_tables is None or table_name.lower() in selected_tables:
                column_names = list(table_info['fields'].keys())
                if selected_columns is not None:
                    cur_selected_columns = [c.lower() for c in column_names 
                                          if f"{table_name}.{c}".lower() in selected_columns]
                else:
                    cur_selected_columns = selected_columns
                output.append(self.single_table_mschema(table_name, cur_selected_columns, 
                                                      example_num, show_type_detail))

        if self.foreign_keys:
            output.append("【Foreign keys】")
            for fk in self.foreign_keys:
                table1, column1, ref_schema, table2, column2 = fk
                if selected_tables is None or \
                        (table1.lower() in selected_tables and table2.lower() in selected_tables):
                    if ref_schema == self.schema:
                        output.append(f"{fk[0]}.{fk[1]}={fk[3]}.{fk[4]}")

        return '\n'.join(output)

    def dump(self) -> Dict:

        return {
            "db_id": self.db_id,
            "schema": self.schema,
            "tables": self.tables,
            "foreign_keys": self.foreign_keys
        }

    def save(self, file_path: str):

        schema_dict = self.dump()
        write_json(file_path, schema_dict)

    def load(self, file_path: str):

        data = read_json(file_path)
        self.db_id = data.get("db_id", "Anonymous")
        self.schema = data.get("schema", None)
        self.tables = data.get("tables", {})
        self.foreign_keys = data.get("foreign_keys", [])


class SchemaEngine:
    
    def __init__(self, engine: Engine, schema: Optional[str] = None, 
                 db_name: Optional[str] = '', sample_rows: int = 5):

        self.engine = engine
        self.schema = schema
        self.db_name = db_name
        self.sample_rows = sample_rows
        
        self.inspector = inspect(engine)
        self._usable_tables = self._get_usable_tables()
        
        self.mschema = MSchema(db_id=db_name, schema=schema)
        self._build_mschema()
    
    def _get_usable_tables(self) -> List[str]:

        if self.schema:
            return [table for table in self.inspector.get_table_names(schema=self.schema)]
        else:
            all_tables = []
            for schema_name in self.inspector.get_schema_names():
                tables = self.inspector.get_table_names(schema=schema_name)
                all_tables.extend(tables)
            return all_tables
    
    def _get_pk_constraint(self, table_name: str) -> List[str]:

        try:
            pk_info = self.inspector.get_pk_constraint(table_name, schema=self.schema)
            return pk_info.get('constrained_columns', [])
        except Exception:
            return []
    
    def _get_table_comment(self, table_name: str) -> str:

        try:
            comment_info = self.inspector.get_table_comment(table_name, schema=self.schema)
            return comment_info.get('text', '') or ''
        except Exception:
            return ''
    
    def _get_foreign_keys(self, table_name: str) -> List[Dict]:

        try:
            return self.inspector.get_foreign_keys(table_name, schema=self.schema)
        except Exception:
            return []
    
    def _fetch_distinct_values(self, table_name: str, column_name: str, max_num: int = 5) -> List:

        try:
            if self.schema:
                full_table_name = f"{self.schema}.{table_name}"
            else:
                full_table_name = table_name
            
            query = text(f'SELECT DISTINCT "{column_name}" FROM "{full_table_name}" LIMIT {max_num}')
            
            with self.engine.connect() as connection:
                result = connection.execute(query)
                values = []
                for row in result:
                    if row[0] is not None and row[0] != '':
                        values.append(row[0])
                return values
        except Exception:
            return []
    
    def _build_mschema(self):
        for table_name in self._usable_tables:
            table_comment = self._get_table_comment(table_name)
            table_with_schema = f"{self.schema}.{table_name}" if self.schema else table_name
            
            self.mschema.add_table(table_with_schema, fields={}, comment=table_comment)
            
            primary_keys = self._get_pk_constraint(table_name)
            foreign_keys = self._get_foreign_keys(table_name)
            
            for fk in foreign_keys:
                referred_schema = fk.get('referred_schema', self.schema)
                referred_table = fk['referred_table']
                for local_col, ref_col in zip(fk['constrained_columns'], fk['referred_columns']):
                    self.mschema.add_foreign_key(
                        table_with_schema, local_col, referred_schema, 
                        f"{referred_schema}.{referred_table}" if referred_schema else referred_table, ref_col
                    )
            
            try:
                columns = self.inspector.get_columns(table_name, schema=self.schema)
                
                for column in columns:
                    field_name = column['name']
                    field_type = str(column['type'])
                    is_primary_key = field_name in primary_keys
                    nullable = column.get('nullable', True)
                    default = column.get('default', None)
                    comment = column.get('comment', '') or ''
                    autoincrement = column.get('autoincrement', False)
                    
                    examples = self._fetch_distinct_values(table_name, field_name, self.sample_rows)
                    examples = examples_to_str(examples)
                    
                    self.mschema.add_field(
                        table_with_schema,
                        field_name,
                        field_type=field_type,
                        primary_key=is_primary_key,
                        nullable=nullable,
                        default=default,
                        autoincrement=autoincrement,
                        comment=comment,
                        examples=examples
                    )
                    
            except Exception as e:
                pass
    
    def get_mschema_string(self, selected_tables: List[str] = None, 
                          selected_columns: List[str] = None,
                          example_num: int = 3, 
                          show_type_detail: bool = False) -> str:

        return self.mschema.to_mschema(
            selected_tables=selected_tables,
            selected_columns=selected_columns,
            example_num=example_num,
            show_type_detail=show_type_detail
        )
    
    def save_mschema(self, file_path: str):

        self.mschema.save(file_path)


class SchemaHandler:
    
    def __init__(self):
        self._schema_cache = {}
    
    def get_database_info(self, db_path: str, db_id: str) -> Dict:

        if db_id in self._schema_cache:
            return self._schema_cache[db_id]
        
        abs_path = os.path.abspath(db_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Database file not found: {abs_path}")
        
        engine = create_engine(f'sqlite:///{abs_path}')
        schema_engine = SchemaEngine(engine=engine, db_name=db_id)
        description = self._load_database_description(db_path, db_id, engine)
        csv_descriptions_only = self._load_csv_descriptions_only(db_path, db_id)
        mschema_str = schema_engine.get_mschema_string(example_num=3)
        
        db_info = {
            'db_id': db_id,
            'db_path': abs_path,
            'description': description,
            'csv_descriptions_only': csv_descriptions_only,
            'mschema': mschema_str,
            'dialect': engine.dialect.name,
            'schema_engine': schema_engine
        }
        
        self._schema_cache[db_id] = db_info
        return db_info
    
    def get_csv_descriptions_only(self, db_path: str, db_id: str) -> str:

        return self._load_csv_descriptions_only(db_path, db_id)
    
    def _generate_ddl(self, engine: Engine) -> Dict[str, str]:

        ddl_statements = {}
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            
            for table_name in tables:
                result = conn.execute(text(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
                ddl = result.fetchone()
                if ddl:
                    ddl_statements[table_name] = ddl[0]
        
        return ddl_statements
    
    def _load_database_description(self, db_path: str, db_id: str, engine: Engine) -> str:

        db_dir = Path(db_path).parent
        desc_dir = db_dir / "database_description"
        
        ddl_statements = self._generate_ddl(engine)
        
        description_parts = []
        description_parts.append(f"Database: {db_id}")
        description_parts.append("=" * 60)
        
        table_descriptions = {}
        if desc_dir.exists():
            for csv_file in desc_dir.glob("*.csv"):
                table_name = csv_file.stem
                try:
                    content = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(csv_file, 'r', encoding=encoding) as f:
                                content = f.read().strip()
                                break
                        except UnicodeDecodeError:
                            continue
                    
                    if content:
                        table_descriptions[table_name] = content
                    else:
                        table_descriptions[table_name] = "Description file found but could not be read due to encoding issues"
                except Exception:
                    table_descriptions[table_name] = "Description file found but could not be read"
        
        for table_name, ddl in ddl_statements.items():
            description_parts.append(f"\nTable: {table_name}")
            description_parts.append("-" * 40)
            
            description_parts.append("DDL:")
            description_parts.append(ddl)
            
            if table_name in table_descriptions:
                description_parts.append("\nDescription:")
                description_parts.append(table_descriptions[table_name])
            else:
                description_parts.append("\nDescription: No description available")
        
        if not ddl_statements:
            description_parts.append("No tables found in database")
        
        return "\n".join(description_parts)
    
    def _load_csv_descriptions_only(self, db_path: str, db_id: str) -> str:

        db_dir = Path(db_path).parent / 'database_description'
        
        table_descriptions = {}
        if db_dir.exists():
            for csv_file in db_dir.glob("*.csv"):
                table_name = csv_file.stem
                try:
                    content = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(csv_file, 'r', encoding=encoding) as f:
                                content = f.read().strip()
                                break
                        except UnicodeDecodeError:
                            continue
                    
                    if content:
                        table_descriptions[table_name] = content
                except Exception as e:
                    continue
        
        description_parts = []
        if table_descriptions:
            description_parts.append("Table Descriptions from CSV files:")
            description_parts.append("=" * 50)
            
            for table_name, description in table_descriptions.items():
                description_parts.append(f"\nTable: {table_name}")
                description_parts.append("-" * 30)
                description_parts.append(description)
        else:
            description_parts.append("No CSV description files found for this database.")
        
        return "\n".join(description_parts)
