import re
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import sqlite3

from tools.schema_representations.mschema import SchemaHandler


class HybridTableColumnSchemaLinker:
    
    def __init__(self, embedding_model, config):

        self.embedding_model = embedding_model
        self.config = config
        self.phase_config = config.get_phase_config(1)
        
        self.top_k_tables = self.phase_config.get('top_k_tables', 10)
        self.top_k_columns_per_table = self.phase_config.get('top_k_columns_per_table', 5)
        self.include_join_tables = self.phase_config.get('include_join_tables', True)
        self.max_join_depth = self.phase_config.get('max_join_depth', 2)
        self.enable_value_linking = self.phase_config.get('enable_value_linking', True)
        self.value_similarity_threshold = self.phase_config.get('value_similarity_threshold', 0.8)
    
    def link_schema(self, question: str, database_info: Dict[str, Any]) -> Dict[str, Any]:

        schema_engine = database_info['schema_engine']
        tables = self._extract_table_info(schema_engine)
        columns = self._extract_column_info(schema_engine)
        
        selected_tables = self._link_tables(question, tables)
        
        selected_columns = self._link_columns(question, columns, selected_tables)
        
        if self.enable_value_linking:
            value_linked_elements = self._link_values(question, database_info, selected_tables)
            selected_tables.update(value_linked_elements['tables'])
            selected_columns.update(value_linked_elements['columns'])
        
        if self.include_join_tables:
            expanded_elements = self._expand_for_joins(
                selected_tables, selected_columns, schema_engine
            )
            selected_tables.update(expanded_elements['tables'])
            selected_columns.update(expanded_elements['columns'])
        
        result = {
            'selected_tables': list(selected_tables.keys()),
            'selected_columns': list(selected_columns.keys()),
            'table_scores': {table: info['score'] for table, info in selected_tables.items()},
            'column_scores': {col: info['score'] for col, info in selected_columns.items()},
            'linking_details': {
                'table_linking': selected_tables,
                'column_linking': selected_columns,
                'value_linking_used': self.enable_value_linking,
                'join_expansion_used': self.include_join_tables
            }
        }
        
        return result
    
    def _extract_table_info(self, schema_engine) -> Dict[str, Dict[str, Any]]:

        tables = {}
        
        for table_name in schema_engine._usable_tables:
            comment = schema_engine._get_table_comment(table_name)
            
            tables[table_name] = {
                'name': table_name,
                'comment': comment,
                'searchable_text': f"{table_name} {comment}".lower(),
                'columns': []
            }
        
        return tables
    
    def _extract_column_info(self, schema_engine) -> Dict[str, Dict[str, Any]]:

        columns = {}
        
        for table_name in schema_engine._usable_tables:
            try:
                table_columns = schema_engine.inspector.get_columns(table_name)
                
                for column in table_columns:
                    column_name = column['name']
                    column_key = f"{table_name}.{column_name}"
                    
                    comment = column.get('comment', '') or ''
                    column_type = str(column['type'])
                    
                    examples = schema_engine._fetch_distinct_values(table_name, column_name)
                    examples_str = ' '.join(str(ex) for ex in examples[:5])
                    
                    columns[column_key] = {
                        'table': table_name,
                        'name': column_name,
                        'type': column_type,
                        'comment': comment,
                        'examples': examples,
                        'searchable_text': f"{column_name} {comment} {examples_str}".lower()
                    }
                    
            except Exception as e:
                continue
        
        return columns
    
    def _link_tables(self, question: str, tables: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

        if not tables:
            return {}
        
        question_text = question.lower()
        table_texts = [info['searchable_text'] for info in tables.values()]
        table_names = list(tables.keys())
        
        all_texts = [question_text] + table_texts
        embeddings = self.embedding_model.encode(all_texts)
        
        question_embedding = embeddings[0]
        table_embeddings = embeddings[1:]
        
        similarities = torch.cosine_similarity(
            question_embedding.unsqueeze(0),
            table_embeddings,
            dim=1
        ).numpy()
        
        selected_tables = {}
        top_indices = np.argsort(similarities)[::-1][:self.top_k_tables]
        
        for idx in top_indices:
            table_name = table_names[idx]
            score = float(similarities[idx])
            
            selected_tables[table_name] = {
                **tables[table_name],
                'score': score,
                'selection_reason': 'embedding_similarity'
            }
        
        return selected_tables
    
    def _link_columns(self, question: str, columns: Dict[str, Dict[str, Any]], 
                     selected_tables: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

        relevant_columns = {
            col_key: col_info for col_key, col_info in columns.items()
            if col_info['table'] in selected_tables
        }
        
        if not relevant_columns:
            return {}
        
        question_text = question.lower()
        column_texts = [info['searchable_text'] for info in relevant_columns.values()]
        column_keys = list(relevant_columns.keys())
        
        all_texts = [question_text] + column_texts
        embeddings = self.embedding_model.encode(all_texts)
        
        question_embedding = embeddings[0]
        column_embeddings = embeddings[1:]
        
        similarities = torch.cosine_similarity(
            question_embedding.unsqueeze(0),
            column_embeddings,
            dim=1
        ).numpy()
        
        selected_columns = {}
        
        columns_by_table = defaultdict(list)
        for i, col_key in enumerate(column_keys):
            table_name = relevant_columns[col_key]['table']
            columns_by_table[table_name].append((col_key, similarities[i]))
        
        for table_name, table_columns in columns_by_table.items():
            table_columns.sort(key=lambda x: x[1], reverse=True)
            
            for col_key, score in table_columns[:self.top_k_columns_per_table]:
                selected_columns[col_key] = {
                    **relevant_columns[col_key],
                    'score': float(score),
                    'selection_reason': 'embedding_similarity'
                }
        
        return selected_columns
    
    def _link_values(self, question: str, database_info: Dict[str, Any], 
                    selected_tables: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

        result = {'tables': {}, 'columns': {}}
        
        try:
            conn = sqlite3.connect(database_info['db_path'])
            cursor = conn.cursor()
            
            potential_values = self._extract_question_values(question)
            
            for value in potential_values:
                schema_engine = database_info['schema_engine']
                
                for table_name in schema_engine._usable_tables:
                    try:
                        columns = schema_engine.inspector.get_columns(table_name)
                        
                        for column in columns:
                            column_name = column['name']
                            column_key = f"{table_name}.{column_name}"
                            
                            try:
                                query = f'SELECT COUNT(*) FROM "{table_name}" WHERE LOWER(CAST("{column_name}" AS TEXT)) LIKE ?'
                                cursor.execute(query, [f'%{value.lower()}%'])
                                count = cursor.fetchone()[0]
                                
                                if count > 0:
                                    if table_name not in selected_tables:
                                        result['tables'][table_name] = {
                                            'name': table_name,
                                            'score': self.value_similarity_threshold,
                                            'selection_reason': f'value_match_{value}'
                                        }
                                    
                                    result['columns'][column_key] = {
                                        'table': table_name,
                                        'name': column_name,
                                        'score': self.value_similarity_threshold,
                                        'selection_reason': f'value_match_{value}',
                                        'matched_value': value
                                    }
                                    
                            except Exception:
                                continue
                                
                    except Exception:
                        continue
            
            conn.close()
            
        except Exception as e:
            pass
        
        return result
    
    def _extract_question_values(self, question: str) -> List[str]:

        values = []
        
        quoted_pattern = r"['\"]([^'\"]+)['\"]"
        quoted_matches = re.findall(quoted_pattern, question)
        values.extend(quoted_matches)
        
        word_pattern = r'\b[A-Z][a-z]+\b'
        capitalized_words = re.findall(word_pattern, question)
        values.extend(capitalized_words)
        
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, question)
        values.extend(numbers)
        
        stop_words = {'The', 'This', 'That', 'What', 'Which', 'Where', 'When', 'How', 'Why'}
        values = [v for v in values if v not in stop_words]
        
        return values
    
    def _expand_for_joins(self, selected_tables: Dict[str, Dict[str, Any]], 
                         selected_columns: Dict[str, Dict[str, Any]], 
                         schema_engine) -> Dict[str, Dict[str, Any]]:

        result = {'tables': {}, 'columns': {}}
        
        all_fks = []
        for table_name in schema_engine._usable_tables:
            fks = schema_engine._get_foreign_keys(table_name)
            for fk in fks:
                all_fks.append({
                    'source_table': table_name,
                    'source_columns': fk['constrained_columns'],
                    'target_table': fk['referred_table'],
                    'target_columns': fk['referred_columns']
                })
        
        selected_table_names = set(selected_tables.keys())
        
        for fk in all_fks:
            source_table = fk['source_table']
            target_table = fk['target_table']
            
            if source_table in selected_table_names and target_table not in selected_table_names:
                needs_join = False
                for col_key in selected_columns:
                    table, column = col_key.split('.')
                    if table == source_table and column in fk['source_columns']:
                        needs_join = True
                        break
                
                if needs_join:
                    result['tables'][target_table] = {
                        'name': target_table,
                        'score': 0.7,
                        'selection_reason': f'join_expansion_from_{source_table}'
                    }
                    
                    pk_columns = schema_engine._get_pk_constraint(target_table)
                    for pk_col in pk_columns:
                        col_key = f"{target_table}.{pk_col}"
                        result['columns'][col_key] = {
                            'table': target_table,
                            'name': pk_col,
                            'score': 0.7,
                            'selection_reason': f'join_expansion_pk'
                        }
            
            elif target_table in selected_table_names and source_table not in selected_table_names:
                needs_join = False
                for col_key in selected_columns:
                    table, column = col_key.split('.')
                    if table == target_table and column in fk['target_columns']:
                        needs_join = True
                        break
                
                if needs_join:
                    result['tables'][source_table] = {
                        'name': source_table,
                        'score': 0.7,
                        'selection_reason': f'join_expansion_from_{target_table}'
                    }
                    
                    for fk_col in fk['source_columns']:
                        col_key = f"{source_table}.{fk_col}"
                        result['columns'][col_key] = {
                            'table': source_table,
                            'name': fk_col,
                            'score': 0.7,
                            'selection_reason': f'join_expansion_fk'
                        }
        
        return result


class LLMSchemaRefiner:
    
    def __init__(self, model_manager, config):

        self.model_manager = model_manager
        self.config = config
        self.phase_config = config.get_phase_config(1)
        
        self.enable_refinement = self.phase_config.get('enable_llm_refinement', True)
        self.refinement_model = self.phase_config.get('refinement_model', 'XGenerationLab/XiYanSQL-QwenCoder-32B-2504')
        self.refinement_model_type = self.phase_config.get('refinement_model_type', 'local')
        self.refinement_model_path = self.phase_config.get('refinement_model_path', None)
        self.max_refinement_tokens = self.phase_config.get('max_refinement_tokens', 2000)
        self.use_json_output = self.phase_config.get('use_json_output', True)
        
        self.include_database_descriptions = self.phase_config.get('include_database_descriptions', True)
        self.max_description_length = self.phase_config.get('max_description_length', 8000)
        
        self.logger = None
    
    def set_logger(self, logger):

        self.logger = logger
    
    def refine_schema(self, question: str, evidence: str, initial_selection: Dict[str, Any], 
                     database_info: Dict[str, Any]) -> Dict[str, Any]:

        if not self.enable_refinement:
            return initial_selection
        
        question_id = initial_selection.get('question_id', 'unknown')
        prompt = None
        response = None
        model = None
        json_mode = False
        
        try:
            model = self.model_manager.load_refinement_model(
                self.refinement_model, self.refinement_model_type, self.refinement_model_path
            )
            
            prompt = self._build_refinement_prompt(
                question, evidence, initial_selection, database_info
            )
            
            if self.logger:
                self.logger.save_prompt('phase1_schema_refinement', prompt, question_id)
            
            json_mode = True
            
            if self.refinement_model_type == 'commercial':
                response = model.generate(prompt, max_tokens=self.max_refinement_tokens, json_mode=json_mode)
            else:
                response = model.generate(prompt, max_new_tokens=self.max_refinement_tokens, json_mode=json_mode)
            
            refined_selection = self._parse_refinement_response(response, initial_selection, json_mode)
            
            if self.logger:
                output_format = 'json' if json_mode else 'text'
                parsing_success = refined_selection.get('refinement_used', False)
                
                self.logger.log_model_output(
                    question_id=question_id,
                    phase='phase1',
                    model_purpose='schema_refinement',
                    model_name=self.refinement_model,
                    model_type=self.refinement_model_type,
                    prompt=prompt,
                    raw_output=response,
                    parsed_output={
                        'selected_tables': refined_selection.get('selected_tables', []),
                        'selected_columns': refined_selection.get('selected_columns', []),
                        'refinement_reasoning': refined_selection.get('refinement_reasoning', ''),
                        'refinement_confidence': refined_selection.get('refinement_confidence', None),
                        'refinement_format': refined_selection.get('refinement_format', 'text'),
                        'success': True
                    },
                    output_format=output_format,
                    parsing_success=parsing_success
                )
            
            return refined_selection
            
        except Exception as e:
            if self.logger and prompt is not None:
                output_format = 'json' if json_mode else 'text'
                error_message = str(e)
                
                self.logger.log_model_output(
                    question_id=question_id,
                    phase='phase1',
                    model_purpose='schema_refinement',
                    model_name=self.refinement_model,
                    model_type=self.refinement_model_type,
                    prompt=prompt,
                    raw_output=response or f"ERROR: {error_message}",
                    parsed_output={
                        'success': False,
                        'error': error_message,
                        'fallback_to_initial_selection': True
                    },
                    output_format=output_format,
                    parsing_success=False
                )
            
            print(f"Warning: Schema refinement failed: {str(e)}")
            return initial_selection
    
    def _build_refinement_prompt(self, question: str, evidence: str, 
                                initial_selection: Dict[str, Any], 
                                database_info: Dict[str, Any]) -> str:
        schema_str = database_info['mschema']
        
        detailed_descriptions = ""
        if self.include_database_descriptions:
            detailed_descriptions = database_info.get('csv_descriptions_only', '')
            
            if not detailed_descriptions.strip():
                detailed_descriptions = "No CSV description files found for this database."
            elif len(detailed_descriptions) > self.max_description_length:
                detailed_descriptions = detailed_descriptions[:self.max_description_length] + "... [truncated for length]"
        
        selected_tables = initial_selection['selected_tables']
        selected_columns = initial_selection['selected_columns']
        
        prompt_parts = [
            "You are a database schema expert. Your task is to refine the initial schema selection for a natural language question.",
            "",
            f"Question: {question}",
            "",
            f"Evidence/Hints: {evidence}",
            "",
            f"Database Schema (M-Schema Format):",
            schema_str
        ]
        
        if self.include_database_descriptions and detailed_descriptions:
            prompt_parts.extend([
                "",
                "Database Table Descriptions (from CSV files):",
                detailed_descriptions
            ])
        
        prompt_parts.extend([
            "",
            "Initial Schema Selection:",
            f"Selected Tables: {', '.join(selected_tables)}",
            f"Selected Columns: {', '.join(selected_columns)}",
            ""
        ])
        
        if self.include_database_descriptions and detailed_descriptions:
            analysis_instructions = """Please analyze the question and refine the schema selection. Use both the schema structure and the detailed descriptions to understand:
1. Are all necessary tables included for answering the question?
2. Are all necessary columns included?
3. Are there any unnecessary tables or columns that should be removed?
4. Are there any missing tables or columns needed for joins?
5. Do the detailed descriptions reveal any additional relevant tables or columns?"""
            reasoning_note = ", referencing specific descriptions when helpful"
        else:
            analysis_instructions = """Please analyze the question and refine the schema selection. Consider:
1. Are all necessary tables included for answering the question?
2. Are all necessary columns included?
3. Are there any unnecessary tables or columns that should be removed?
4. Are there any missing tables or columns needed for joins?"""
            reasoning_note = ""
        
        prompt_parts.extend([
            analysis_instructions,
            ""
        ])
        
        prompt_parts.extend([
            "IMPORTANT: You MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.",
            "",
            "Required JSON format:",
            "",
            "{",
            '  "REFINED_TABLES": ["table1", "table2", ...],',
            '  "REFINED_COLUMNS": ["table1.column1", "table2.column2", ...],',
            f'  "REASONING": "brief explanation of your refinements{reasoning_note}"',
            "}",
            "",
            "Requirements:",
            "- REFINED_TABLES: List of table names to include (no quotes around table names in the list)",
            "- REFINED_COLUMNS: List of column names in 'table.column' format",
            "- REASONING: Brief explanation of your refinement decisions",
            "- Respond with ONLY the JSON object, no additional text",
            "- Ensure the JSON is properly formatted and valid"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        if self.logger:
            descriptions_enabled = self.include_database_descriptions
            has_descriptions = (detailed_descriptions and 
                              detailed_descriptions != "No CSV description files found for this database.")
            desc_length = len(detailed_descriptions) if detailed_descriptions else 0
            was_truncated = "truncated for length" in detailed_descriptions if detailed_descriptions else False
            
            self.logger.debug(
                f"Schema refinement prompt built: csv_descriptions_enabled={descriptions_enabled}, "
                f"csv_descriptions_available={has_descriptions}, description_length={desc_length}, "
                f"truncated={was_truncated}",
                phase='phase1'
            )

        return prompt
    
    def _parse_refinement_response(self, response: str, 
                                  initial_selection: Dict[str, Any], 
                                  json_mode: bool = False) -> Dict[str, Any]:

        try:
            refined_selection = initial_selection.copy()
            
            try:
                import json
                cleaned_response = response.strip()
                
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                if not cleaned_response.startswith('{'):
                    import re
                    json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                    if json_match:
                        cleaned_response = json_match.group(0)
                
                parsed_json = json.loads(cleaned_response)
                
                if 'REFINED_TABLES' in parsed_json:
                    refined_selection['selected_tables'] = parsed_json['REFINED_TABLES']
                elif 'refined_tables' in parsed_json:
                    refined_selection['selected_tables'] = parsed_json['refined_tables']
                
                if 'REFINED_COLUMNS' in parsed_json:
                    refined_selection['selected_columns'] = parsed_json['REFINED_COLUMNS']
                elif 'refined_columns' in parsed_json:
                    refined_selection['selected_columns'] = parsed_json['refined_columns']
                
                if 'REASONING' in parsed_json:
                    refined_selection['refinement_reasoning'] = parsed_json['REASONING']
                elif 'reasoning' in parsed_json:
                    refined_selection['refinement_reasoning'] = parsed_json['reasoning']
                
                if 'confidence' in parsed_json:
                    refined_selection['refinement_confidence'] = parsed_json['confidence']
                
                refined_selection['refinement_used'] = True
                refined_selection['refinement_format'] = 'json'
                return refined_selection
                
            except json.JSONDecodeError:
                print("Warning: JSON parsing failed, falling back to text parsing")
            
            tables_match = re.search(r'REFINED_TABLES:\s*\[(.*?)\]', response, re.DOTALL)
            if tables_match:
                tables_str = tables_match.group(1)
                refined_tables = [t.strip().strip('"\'') for t in tables_str.split(',') if t.strip()]
                refined_selection['selected_tables'] = refined_tables
            
            columns_match = re.search(r'REFINED_COLUMNS:\s*\[(.*?)\]', response, re.DOTALL)
            if columns_match:
                columns_str = columns_match.group(1)
                refined_columns = [c.strip().strip('"\'') for c in columns_str.split(',') if c.strip()]
                refined_selection['selected_columns'] = refined_columns
            
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?:\n\n|\Z)', response, re.DOTALL)
            if reasoning_match:
                refined_selection['refinement_reasoning'] = reasoning_match.group(1).strip()
            
            refined_selection['refinement_used'] = True
            refined_selection['refinement_format'] = 'text'
            return refined_selection
            
        except Exception as e:
            print(f"Warning: Failed to parse refinement response: {str(e)}")
            return initial_selection


class SchemaLinker:
    
    def __init__(self, config, model_manager):

        self.config = config
        self.model_manager = model_manager
        self.phase_config = config.get_phase_config(1)
        
        embedding_model = model_manager.load_embedding_model()
        self.hytcsl = HybridTableColumnSchemaLinker(embedding_model, config)
        self.refiner = LLMSchemaRefiner(model_manager, config)
        
        self.schema_representations = self.phase_config.get('schema_representations', ['m_schema'])
        self.include_examples = self.phase_config.get('include_examples', True)
        self.max_examples_per_column = self.phase_config.get('max_examples_per_column', 3)
        
        self.logger = None
    
    def set_logger(self, logger):
 
        self.logger = logger
        self.refiner.set_logger(logger)
    
    def link_schema(self, question: Dict[str, Any], database_info: Dict[str, Any]) -> Dict[str, Any]:

        question_text = question['question']
        evidence = question.get('evidence', '')
        
        initial_selection = self.hytcsl.link_schema(question_text, database_info)
        
        refined_selection = self.refiner.refine_schema(
            question_text, evidence, initial_selection, database_info
        )
        
        schema_representations = self._generate_schema_representations(
            refined_selection, database_info
        )
        
        result = {
            'question_id': question['question_id'],
            'selected_tables': refined_selection['selected_tables'],
            'selected_columns': refined_selection['selected_columns'],
            'refinement_used': refined_selection.get('refinement_used', False),
            'refinement_reasoning': refined_selection.get('refinement_reasoning', ''),
            'schema_representations': schema_representations,
            'linking_details': {
                'initial_selection': initial_selection,
                'refined_selection': refined_selection,
                'hytcsl_config': {
                    'top_k_tables': self.hytcsl.top_k_tables,
                    'top_k_columns_per_table': self.hytcsl.top_k_columns_per_table,
                    'value_linking_enabled': self.hytcsl.enable_value_linking,
                    'join_expansion_enabled': self.hytcsl.include_join_tables
                }
            }
        }
        
        return result
    
    def _generate_schema_representations(self, selection: Dict[str, Any], 
                                       database_info: Dict[str, Any]) -> Dict[str, str]:

        representations = {}
        schema_engine = database_info['schema_engine']
        
        selected_tables = selection['selected_tables']
        selected_columns = selection['selected_columns']
        
        for repr_type in self.schema_representations:
            if repr_type == 'm_schema':
                representations['m_schema'] = schema_engine.get_mschema_string(
                    selected_tables=selected_tables,
                    selected_columns=selected_columns,
                    example_num=self.max_examples_per_column if self.include_examples else 0
                )
            
            elif repr_type == 'ddl':
                representations['ddl'] = self._generate_ddl_representation(
                    selected_tables, database_info
                )
            
            elif repr_type == 'json':
                representations['json'] = self._generate_json_representation(
                    selected_tables, selected_columns, schema_engine
                )
            
            elif repr_type == 'markdown':
                representations['markdown'] = self._generate_markdown_representation(
                    selected_tables, selected_columns, schema_engine
                )
        
        return representations
    
    def _generate_ddl_representation(self, selected_tables: List[str], 
                                   database_info: Dict[str, Any]) -> str:
        ddl_parts = []
        
        try:
            import sqlite3
            conn = sqlite3.connect(database_info['db_path'])
            cursor = conn.cursor()
            
            for table_name in selected_tables:
                cursor.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                result = cursor.fetchone()
                if result:
                    ddl_parts.append(result[0])
            
            conn.close()
            
        except Exception as e:
            ddl_parts.append(f"-- Error generating DDL: {str(e)}")
        
        return '\n\n'.join(ddl_parts)
    
    def _generate_json_representation(self, selected_tables: List[str], 
                                    selected_columns: List[str], 
                                    schema_engine) -> str:

        import json
        
        schema_dict = {
            'database': schema_engine.db_name,
            'tables': {}
        }
        
        for table_name in selected_tables:
            schema_dict['tables'][table_name] = {
                'name': table_name,
                'comment': schema_engine._get_table_comment(table_name),
                'columns': {}
            }
            
            try:
                columns = schema_engine.inspector.get_columns(table_name)
                for column in columns:
                    column_name = column['name']
                    column_key = f"{table_name}.{column_name}"
                    
                    if column_key in selected_columns:
                        schema_dict['tables'][table_name]['columns'][column_name] = {
                            'name': column_name,
                            'type': str(column['type']),
                            'nullable': column.get('nullable', True),
                            'comment': column.get('comment', ''),
                            'examples': schema_engine._fetch_distinct_values(
                                table_name, column_name, 3
                            ) if self.include_examples else []
                        }
            except Exception:
                continue
        
        return json.dumps(schema_dict, indent=2)
    
    def _generate_markdown_representation(self, selected_tables: List[str], 
                                        selected_columns: List[str], 
                                        schema_engine) -> str:
        markdown_parts = [f"# Database Schema: {schema_engine.db_name}", ""]
        
        for table_name in selected_tables:
            comment = schema_engine._get_table_comment(table_name)
            markdown_parts.append(f"## Table: {table_name}")
            if comment:
                markdown_parts.append(f"*{comment}*")
            markdown_parts.append("")
            
            markdown_parts.append("| Column | Type | Comment | Examples |")
            markdown_parts.append("|--------|------|---------|----------|")
            
            try:
                columns = schema_engine.inspector.get_columns(table_name)
                for column in columns:
                    column_name = column['name']
                    column_key = f"{table_name}.{column_name}"
                    
                    if column_key in selected_columns:
                        column_type = str(column['type'])
                        column_comment = column.get('comment', '')
                        
                        examples = []
                        if self.include_examples:
                            examples = schema_engine._fetch_distinct_values(
                                table_name, column_name, 3
                            )
                        examples_str = ', '.join(str(ex) for ex in examples[:3])
                        
                        markdown_parts.append(
                            f"| {column_name} | {column_type} | {column_comment} | {examples_str} |"
                        )
            except Exception:
                markdown_parts.append("| *Error loading columns* | | | |")
            
            markdown_parts.append("")
        
        return '\n'.join(markdown_parts)
