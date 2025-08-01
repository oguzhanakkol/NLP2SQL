import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass

import sqlglot
from sqlglot import parse_one


@dataclass
class SelectionCandidate:
    sql: str
    original_candidates: List[Dict[str, Any]]
    
    validity_score: float = 0.0
    popularity_score: float = 0.0
    llm_critic_score: float = 0.0
    value_alignment_score: float = 0.0
    self_consistency_score: float = 0.0
    
    final_score: float = 0.0
    selection_reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sql': self.sql,
            'original_candidates': self.original_candidates,
            'validity_score': self.validity_score,
            'popularity_score': self.popularity_score,
            'llm_critic_score': self.llm_critic_score,
            'value_alignment_score': self.value_alignment_score,
            'self_consistency_score': self.self_consistency_score,
            'final_score': self.final_score,
            'selection_reasoning': self.selection_reasoning
        }


class ValidityFilter:
    
    def __init__(self, config):

        self.config = config
        self.phase_config = config.get_phase_config(3)
        
        self.remove_syntax_errors = self.phase_config.get('remove_syntax_errors', True)
        self.remove_duplicates = self.phase_config.get('remove_duplicates', True)
        self.canonicalization_method = self.phase_config.get('canonicalization_method', 'sqlglot')
    
    def filter_candidates(self, candidates: List[Dict[str, Any]]) -> tuple[List[SelectionCandidate], Dict[str, int]]:

        if self.remove_syntax_errors:
            valid_candidates = [c for c in candidates if c.get('is_valid', True)]
        else:
            valid_candidates = candidates
        
        after_syntax_count = len(valid_candidates)
        
        if not valid_candidates:
            return [], {
                'original_count': len(candidates),
                'after_syntax_filter': after_syntax_count,
                'after_duplicate_filter': 0,
                'final_count': 0
            }
        
        if self.remove_duplicates:
            canonical_groups = self._group_by_canonical_sql(valid_candidates)
        else:
            canonical_groups = {i: [c] for i, c in enumerate(valid_candidates)}
        
        selection_candidates = []
        for canonical_sql, candidate_group in canonical_groups.items():
            if isinstance(canonical_sql, int):
                original_sql = candidate_group[0]['sql']
            else:
                original_sql = candidate_group[0]['sql']
            
            selection_candidate = SelectionCandidate(
                sql=original_sql,
                original_candidates=candidate_group
            )
            
            selection_candidate.validity_score = 1.0
            selection_candidate.popularity_score = len(candidate_group) / len(valid_candidates)
            
            selection_candidates.append(selection_candidate)
        
        filtering_stats = {
            'original_count': len(candidates),
            'after_syntax_filter': after_syntax_count,
            'after_duplicate_filter': len(selection_candidates),
            'final_count': len(selection_candidates)
        }
        
        return selection_candidates, filtering_stats
    
    def _group_by_canonical_sql(self, candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:

        groups = defaultdict(list)
        
        for candidate in candidates:
            sql = candidate['sql']
            canonical_sql = self._canonicalize_sql(sql)
            groups[canonical_sql].append(candidate)
        
        return dict(groups)
    
    def _canonicalize_sql(self, sql: str) -> str:

        if self.canonicalization_method == 'sqlglot':
            return self._canonicalize_with_sqlglot(sql)
        else:
            return self._canonicalize_simple(sql)
    
    def _canonicalize_with_sqlglot(self, sql: str) -> str:

        try:
            parsed = parse_one(sql, dialect='sqlite')
            if parsed:
                canonical = str(parsed)
                canonical = re.sub(r'\s+', ' ', canonical)
                canonical = canonical.upper()
                return canonical.strip()
        except Exception:
            pass
        
        return self._canonicalize_simple(sql)
    
    def _canonicalize_simple(self, sql: str) -> str:
        canonical = re.sub(r'\s+', ' ', sql.strip())
        canonical = canonical.upper()
        
        if canonical.endswith(';'):
            canonical = canonical[:-1].strip()
        
        return canonical


class LLMCritic:
    
    def __init__(self, model_manager, config):

        self.model_manager = model_manager
        self.config = config
        self.phase_config = config.get_phase_config(3)
        
        self.ranking_model = self.phase_config.get('ranking_model', 'XGenerationLab/XiYanSQL-QwenCoder-32B-2504')
        self.ranking_model_type = self.phase_config.get('ranking_model_type', 'local')
        self.ranking_model_path = self.phase_config.get('ranking_model_path', None)
        self.enable_chain_of_thought = self.phase_config.get('enable_chain_of_thought', True)
        self.max_ranking_tokens = self.phase_config.get('max_ranking_tokens', 1500)
        self.use_json_output = self.phase_config.get('use_json_output', True)
    
    def rank_candidates(self, question: str, evidence: str, schema_representation: str,
                       candidates: List[SelectionCandidate], question_id: int = None, logger = None) -> List[SelectionCandidate]:
        if not candidates:
            return candidates
        
        prompt = None
        response = None
        model = None
        json_mode = False
        
        try:
            model = self.model_manager.load_ranking_model(
                self.ranking_model, self.ranking_model_type, self.ranking_model_path
            )
            
            prompt = self._build_ranking_prompt(
                question, evidence, schema_representation, candidates
            )
            
            if logger:
                logger.save_prompt('phase3_sql_ranking', prompt, question_id)
            
            json_mode = self.use_json_output and self.ranking_model_type == 'commercial'
            
            if self.ranking_model_type == 'commercial':
                response = model.generate(prompt, max_tokens=self.max_ranking_tokens, json_mode=json_mode)
            else:
                response = model.generate(prompt, max_new_tokens=self.max_ranking_tokens, json_mode=json_mode)
            
            ranked_candidates = self._parse_ranking_response(response, candidates, json_mode)
            
            if logger:
                output_format = 'json' if json_mode else 'text'
                parsing_success = any(c.llm_critic_score != 0.5 for c in ranked_candidates)
                
                parsed_rankings = []
                for i, candidate in enumerate(ranked_candidates):
                    parsed_rankings.append({
                        'candidate_index': i,
                        'llm_critic_score': candidate.llm_critic_score,
                        'sql_preview': candidate.sql[:100] + "..." if len(candidate.sql) > 100 else candidate.sql
                    })
                
                logger.log_model_output(
                    question_id=question_id or 0,
                    phase='phase3',
                    model_purpose='sql_ranking',
                    model_name=self.ranking_model,
                    model_type=self.ranking_model_type,
                    prompt=prompt,
                    raw_output=response,
                    parsed_output={
                        'rankings': parsed_rankings,
                        'total_candidates': len(candidates),
                        'ranking_format': 'json' if json_mode else 'text',
                        'success': True
                    },
                    output_format=output_format,
                    parsing_success=parsing_success
                )
            
            return ranked_candidates
            
        except Exception as e:
            if logger and prompt is not None:
                output_format = 'json' if json_mode else 'text'
                error_message = str(e)
                
                logger.log_model_output(
                    question_id=question_id or 0,
                    phase='phase3',
                    model_purpose='sql_ranking',
                    model_name=self.ranking_model,
                    model_type=self.ranking_model_type,
                    prompt=prompt,
                    raw_output=response or f"ERROR: {error_message}",
                    parsed_output={
                        'success': False,
                        'error': error_message,
                        'fallback_to_default_scores': True,
                        'total_candidates': len(candidates)
                    },
                    output_format=output_format,
                    parsing_success=False
                )
            
            print(f"Warning: LLM ranking failed: {str(e)}")
            for candidate in candidates:
                candidate.llm_critic_score = 0.5
            return candidates
    
    def _build_ranking_prompt(self, question: str, evidence: str, schema_representation: str,
                             candidates: List[SelectionCandidate]) -> str:
        prompt_parts = [
            "You are an expert SQL analyst. Your task is to rank SQL query candidates based on how well they answer the given natural language question.",
            "",
            f"Question: {question}",
        ]
        
        if evidence:
            prompt_parts.extend([
                f"Evidence/Hints: {evidence}",
                ""
            ])
        
        prompt_parts.extend([
            "Database Schema:",
            schema_representation,
            "",
            "SQL Candidates to rank:"
        ])
        
        for i, candidate in enumerate(candidates):
            prompt_parts.append(f"Candidate {i+1}:")
            prompt_parts.append(f"```sql")
            prompt_parts.append(candidate.sql)
            prompt_parts.append("```")
            prompt_parts.append("")
        
        json_mode = self.use_json_output and self.ranking_model_type == 'commercial'
        
        if json_mode:
            if self.enable_chain_of_thought:
                prompt_parts.extend([
                    "Please analyze each candidate and rank them from best to worst. Consider:",
                    "1. Correctness: Does the SQL correctly answer the question?",
                    "2. Completeness: Does it include all necessary elements?",
                    "3. Efficiency: Is the query well-structured?",
                    "4. Schema adherence: Does it use the correct tables and columns?",
                    "",
                    "Provide your analysis in JSON format:",
                    "",
                    "{",
                    '  "analysis": "Your detailed analysis of each candidate",',
                    '  "rankings": [',
                    '    {',
                    '      "candidate_number": 1,',
                    '      "score": 0.92,',
                    '      "reasoning": "Brief explanation"',
                    '    },',
                    '    {',
                    '      "candidate_number": 2,',
                    '      "score": 0.78,',
                    '      "reasoning": "Brief explanation"',
                    '    }',
                    '  ]',
                    "}"
                ])
            else:
                prompt_parts.extend([
                    "Rank the candidates from best to worst and provide scores (0.0-1.0) in JSON format:",
                    "",
                    "{",
                    '  "rankings": [',
                    '    {"candidate_number": 1, "score": 0.92},',
                    '    {"candidate_number": 2, "score": 0.78}',
                    '  ]',
                    "}"
                ])
        else:
            if self.enable_chain_of_thought:
                prompt_parts.extend([
                    "Please analyze each candidate and rank them from best to worst. Consider:",
                    "1. Correctness: Does the SQL correctly answer the question?",
                    "2. Completeness: Does it include all necessary elements?",
                    "3. Efficiency: Is the query well-structured?",
                    "4. Schema adherence: Does it use the correct tables and columns?",
                    "",
                    "Provide your analysis in the following format:",
                    "",
                    "ANALYSIS:",
                    "[Your detailed analysis of each candidate]",
                    "",
                    "RANKING:",
                    "1. Candidate X (score: 0.0-1.0) - Brief reason",
                    "2. Candidate Y (score: 0.0-1.0) - Brief reason",
                    "..."
                ])
            else:
                prompt_parts.extend([
                    "Rank the candidates from best to worst and provide scores (0.0-1.0):",
                    "",
                    "RANKING:",
                    "1. Candidate X (score: 0.0-1.0)",
                    "2. Candidate Y (score: 0.0-1.0)",
                    "..."
                ])
        
        return '\n'.join(prompt_parts)
    
    def _parse_ranking_response(self, response: str, candidates: List[SelectionCandidate], json_mode: bool = False) -> List[SelectionCandidate]:

        try:
            for candidate in candidates:
                candidate.llm_critic_score = 0.5
            
            if json_mode:
                try:
                    import json

                    cleaned_response = response.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                    cleaned_response = cleaned_response.strip()
                    
                    parsed_json = json.loads(cleaned_response)
                    
                    if 'rankings' in parsed_json:
                        for ranking in parsed_json['rankings']:
                            candidate_num = ranking.get('candidate_number', 0) - 1
                            score = ranking.get('score', 0.5)
                            
                            if 0 <= candidate_num < len(candidates) and 0.0 <= score <= 1.0:
                                candidates[candidate_num].llm_critic_score = score
                    
                    return candidates
                    
                except json.JSONDecodeError:
                    print("Warning: JSON parsing failed, falling back to text parsing")
            
            ranking_match = re.search(r'RANKING:\s*(.*?)(?:\n\n|\Z)', response, re.DOTALL)
            if not ranking_match:
                ranking_text = response
            else:
                ranking_text = ranking_match.group(1)
            
            ranking_lines = ranking_text.strip().split('\n')
            
            for line in ranking_lines:
                line = line.strip()
                if not line or not re.match(r'^\d+\.', line):
                    continue
                
                candidate_match = re.search(r'Candidate (\d+)', line, re.IGNORECASE)
                score_match = re.search(r'score:\s*([0-9.]+)', line, re.IGNORECASE)
                
                if candidate_match and score_match:
                    candidate_num = int(candidate_match.group(1)) - 1
                    score = float(score_match.group(1))
                    
                    if 0 <= candidate_num < len(candidates) and 0.0 <= score <= 1.0:
                        candidates[candidate_num].llm_critic_score = score
            
        except Exception as e:
            print(f"Warning: Failed to parse LLM ranking: {str(e)}")
        
        return candidates


class ValueAlignmentChecker:
    
    def __init__(self, config):

        self.config = config
        self.phase_config = config.get_phase_config(3)
        
        self.enable_value_alignment = self.phase_config.get('enable_value_alignment', True)
        self.alignment_threshold = self.phase_config.get('value_alignment_threshold', 0.9)
    
    def check_alignment(self, question: str, candidates: List[SelectionCandidate]) -> List[SelectionCandidate]:

        if not self.enable_value_alignment:
            for candidate in candidates:
                candidate.value_alignment_score = 1.0
            return candidates
        
        question_values = self._extract_question_values(question)
        
        if not question_values:
            for candidate in candidates:
                candidate.value_alignment_score = 1.0
            return candidates
        
        for candidate in candidates:
            alignment_score = self._calculate_alignment_score(candidate.sql, question_values)
            candidate.value_alignment_score = alignment_score
        
        return candidates
    
    def _extract_question_values(self, question: str) -> List[str]:

        values = []
        
        quoted_pattern = r"['\"]([^'\"]+)['\"]"
        quoted_matches = re.findall(quoted_pattern, question)
        values.extend(quoted_matches)
        
        number_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(number_pattern, question)
        values.extend(numbers)
        
        entity_pattern = r'\b[A-Z][a-zA-Z]+\b'
        entities = re.findall(entity_pattern, question)
        question_words = {'What', 'Which', 'Where', 'When', 'How', 'Why', 'Who', 'The', 'This', 'That'}
        entities = [e for e in entities if e not in question_words]
        values.extend(entities)
        
        return values
    
    def _calculate_alignment_score(self, sql: str, question_values: List[str]) -> float:

        if not question_values:
            return 1.0
        
        sql_lower = sql.lower()
        matched_values = 0
        
        for value in question_values:
            value_lower = value.lower()
            if value_lower in sql_lower:
                matched_values += 1
        
        return matched_values / len(question_values)


class SelfConsistencyChecker:
    
    def __init__(self, config):
        self.config = config
        self.phase_config = config.get_phase_config(3)
        
        self.enable_self_consistency = self.phase_config.get('enable_self_consistency', True)
        self.consistency_threshold = self.phase_config.get('consistency_threshold', 0.7)
        self.enable_majority_voting = self.phase_config.get('enable_majority_voting', True)
    
    def check_consistency(self, candidates: List[SelectionCandidate]) -> List[SelectionCandidate]:
        if not self.enable_self_consistency:
            for candidate in candidates:
                candidate.self_consistency_score = 1.0
            return candidates
        
        for candidate in candidates:
            consistency_score = self._calculate_consistency_score(candidate, candidates)
            candidate.self_consistency_score = consistency_score
        
        return candidates
    
    def _calculate_consistency_score(self, candidate: SelectionCandidate, 
                                   all_candidates: List[SelectionCandidate]) -> float:
        
        original_candidates = candidate.original_candidates
        
        unique_models = set(c['model_name'] for c in original_candidates)
        model_diversity_score = len(unique_models) / max(1, len(set(
            c['model_name'] for all_c in all_candidates for c in all_c.original_candidates
        )))
        
        unique_configs = set(
            f"{c['model_name']}_{c['schema_representation']}_{c['temperature']}"
            for c in original_candidates
        )
        config_diversity_score = len(unique_configs) / max(1, len(original_candidates))
        
        consistency_score = (
            0.4 * candidate.popularity_score +  # How popular this SQL is
            0.3 * model_diversity_score +       # How many different models agree
            0.3 * config_diversity_score        # How diverse the configurations are
        )
        
        return min(1.0, consistency_score)


class SQLSelector:
    
    def __init__(self, config, model_manager):

        self.config = config
        self.model_manager = model_manager
        self.phase_config = config.get_phase_config(3)
        self.logger = None
        
        self.validity_filter = ValidityFilter(config)
        self.llm_critic = LLMCritic(model_manager, config)
        self.value_checker = ValueAlignmentChecker(config)
        self.consistency_checker = SelfConsistencyChecker(config)
        
        self.selection_weights = {
            'validity': 0.2,
            'popularity': 0.15,
            'llm_critic': 0.35,
            'value_alignment': 0.15,
            'self_consistency': 0.15
        }
    
    def set_logger(self, logger):

        self.logger = logger
    
    def select_best_sql(self, question: Dict[str, Any], candidates_result: Dict[str, Any]) -> Dict[str, Any]:

        question_text = question['question']
        evidence = question.get('evidence', '')
        candidates = candidates_result['candidates']
        
        if self.logger:
            self.logger.log_sql_selection_start(
                question['question_id'], 
                len(candidates), 
                'multi_criteria_ranking'
            )
        
        if not candidates:
            return {
                'question_id': question['question_id'],
                'selected_sql': '',
                'selection_method': 'no_candidates',
                'confidence_score': 0.0,
                'candidates_considered': 0,
                'selection_details': {'error': 'No candidates available'}
            }
        
        if self.logger:
            self.logger.log_selection_process_detail(
                "Filtering", "Removing invalid SQL and duplicates", len(candidates)
            )
        
        original_count = len(candidates)
        selection_candidates, filtering_stats = self.validity_filter.filter_candidates(candidates)
        
        if self.logger:
            self.logger.log_candidate_filtering(
                filtering_stats['original_count'], 
                filtering_stats['after_syntax_filter'],
                filtering_stats['after_duplicate_filter'],
                filtering_stats['final_count']
            )
        
        if not selection_candidates:
            return {
                'question_id': question['question_id'],
                'selected_sql': '',
                'selection_method': 'all_invalid',
                'confidence_score': 0.0,
                'candidates_considered': len(candidates),
                'selection_details': {'error': 'All candidates were invalid'}
            }
        
        schema_representations = candidates_result.get('schema_representations', {})
        schema_repr = (schema_representations.get('m_schema') or 
                      schema_representations.get('ddl') or 
                      schema_representations.get('json') or 
                      schema_representations.get('markdown') or 
                      'Schema not available')
        
        if self.logger:
            self.logger.log_selection_process_detail(
                "LLM Ranking", f"Applying {self.llm_critic.ranking_model} critic scoring", len(selection_candidates)
            )
        
        selection_candidates = self.llm_critic.rank_candidates(
            question_text, evidence, schema_repr, selection_candidates, 
            question['question_id'], self.logger
        )
        
        if self.logger:
            self.logger.log_selection_process_detail(
                "Value Alignment", "Checking question value alignment", len(selection_candidates)
            )
        
        selection_candidates = self.value_checker.check_alignment(
            question_text, selection_candidates
        )
        
        if self.logger:
            self.logger.log_selection_process_detail(
                "Self-Consistency", "Evaluating candidate consistency", len(selection_candidates)
            )
        
        selection_candidates = self.consistency_checker.check_consistency(selection_candidates)
        
        if self.logger:
            self.logger.log_selection_process_detail(
                "Final Scoring", "Computing weighted final scores", len(selection_candidates)
            )
        
        for candidate in selection_candidates:
            final_score = (
                self.selection_weights['validity'] * candidate.validity_score +
                self.selection_weights['popularity'] * candidate.popularity_score +
                self.selection_weights['llm_critic'] * candidate.llm_critic_score +
                self.selection_weights['value_alignment'] * candidate.value_alignment_score +
                self.selection_weights['self_consistency'] * candidate.self_consistency_score
            )
            candidate.final_score = final_score
        
        best_candidate = max(selection_candidates, key=lambda c: c.final_score)
        
        selected_candidate_info = self._get_selected_candidate_info(best_candidate, candidates)
        
        reasoning = self._generate_selection_reasoning(best_candidate, selection_candidates)
        best_candidate.selection_reasoning = reasoning
        
        if self.logger:
            self.logger.log_selected_candidate(
                question['question_id'],
                selected_candidate_info['candidate_number'],
                best_candidate.sql,
                best_candidate.final_score,
                'multi_criteria_ranking',
                selected_candidate_info.get('model_name', ''),
                selected_candidate_info.get('schema_representation', ''),
                selected_candidate_info.get('temperature', 0.0),
                selected_candidate_info.get('original_validity', True),
                selected_candidate_info.get('was_fixed', False)
            )
        
        result = {
            'question_id': question['question_id'],
            'selected_sql': best_candidate.sql,
            'selection_method': 'multi_criteria_ranking',
            'confidence_score': best_candidate.final_score,
            'candidates_considered': len(candidates),
            'valid_candidates': len(selection_candidates),
            'selection_details': {
                'best_candidate': best_candidate.to_dict(),
                'all_candidates': [c.to_dict() for c in selection_candidates],
                'selection_weights': self.selection_weights,
                'selection_reasoning': reasoning,
                'selected_candidate_info': selected_candidate_info
            }
        }
        
        return result
    
    def _get_selected_candidate_info(self, best_candidate: SelectionCandidate, 
                                   original_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        for i, orig_candidate in enumerate(original_candidates, 1):
            if orig_candidate['sql'].strip() == best_candidate.sql.strip():
                return {
                    'candidate_number': i,
                    'model_name': orig_candidate.get('model_name', ''),
                    'schema_representation': orig_candidate.get('schema_representation', ''),
                    'temperature': orig_candidate.get('temperature', 0.0),
                    'original_validity': orig_candidate.get('is_valid', True),
                    'was_fixed': orig_candidate.get('metadata', {}).get('was_fixed', False),
                    'generation_time': orig_candidate.get('generation_time', 0.0)
                }
        
        for i, orig_candidate in enumerate(original_candidates, 1):
            metadata = orig_candidate.get('metadata', {})
            if metadata.get('was_fixed', False) and metadata.get('original_sql', '').strip() == best_candidate.sql.strip():
                return {
                    'candidate_number': i,
                    'model_name': orig_candidate.get('model_name', ''),
                    'schema_representation': orig_candidate.get('schema_representation', ''),
                    'temperature': orig_candidate.get('temperature', 0.0),
                    'original_validity': False,
                    'was_fixed': True,
                    'generation_time': orig_candidate.get('generation_time', 0.0)
                }
        
        if best_candidate.original_candidates:
            first_orig = best_candidate.original_candidates[0]
            for i, orig_candidate in enumerate(original_candidates, 1):
                if (orig_candidate.get('model_name') == first_orig.get('model_name') and
                    orig_candidate.get('schema_representation') == first_orig.get('schema_representation') and
                    orig_candidate.get('temperature') == first_orig.get('temperature')):
                    return {
                        'candidate_number': i,
                        'model_name': orig_candidate.get('model_name', ''),
                        'schema_representation': orig_candidate.get('schema_representation', ''),
                        'temperature': orig_candidate.get('temperature', 0.0),
                        'original_validity': orig_candidate.get('is_valid', True),
                        'was_fixed': orig_candidate.get('metadata', {}).get('was_fixed', False),
                        'generation_time': orig_candidate.get('generation_time', 0.0)
                    }
        
        if best_candidate.original_candidates:
            first_orig = best_candidate.original_candidates[0]
            return {
                'candidate_number': 1,
                'model_name': first_orig.get('model_name', 'Unknown'),
                'schema_representation': first_orig.get('schema_representation', 'Unknown'),
                'temperature': first_orig.get('temperature', 0.0),
                'original_validity': first_orig.get('is_valid', True),
                'was_fixed': first_orig.get('metadata', {}).get('was_fixed', False),
                'generation_time': first_orig.get('generation_time', 0.0)
            }
        
        return {
            'candidate_number': 1,
            'model_name': 'Unknown',
            'schema_representation': 'Unknown',
            'temperature': 0.0,
            'original_validity': True,
            'was_fixed': False,
            'generation_time': 0.0
        }
    
    def _generate_selection_reasoning(self, best_candidate: SelectionCandidate, 
                                    all_candidates: List[SelectionCandidate]) -> str:
        reasoning_parts = [
            f"Selected SQL with final score: {best_candidate.final_score:.3f}"
        ]
        
        reasoning_parts.append("Score breakdown:")
        reasoning_parts.append(f"- Validity: {best_candidate.validity_score:.3f}")
        reasoning_parts.append(f"- Popularity: {best_candidate.popularity_score:.3f}")
        reasoning_parts.append(f"- LLM Critic: {best_candidate.llm_critic_score:.3f}")
        reasoning_parts.append(f"- Value Alignment: {best_candidate.value_alignment_score:.3f}")
        reasoning_parts.append(f"- Self Consistency: {best_candidate.self_consistency_score:.3f}")
        
        reasoning_parts.append(f"Generated by {len(best_candidate.original_candidates)} model configurations")
        
        if len(all_candidates) > 1:
            scores = [c.final_score for c in all_candidates]
            scores.sort(reverse=True)
            if len(scores) > 1:
                reasoning_parts.append(f"Outperformed {len(scores)-1} other candidates (next best: {scores[1]:.3f})")
        
        return '\n'.join(reasoning_parts)
