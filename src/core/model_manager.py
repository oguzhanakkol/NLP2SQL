import os
import torch
from typing import Dict, List, Optional, Any, Union
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import tiktoken
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ModelUsage:
    input_tokens: int
    output_tokens: int
    cost: Optional[float] = None
    model_name: str = ""
    timestamp: str = ""


class BaseModel(ABC):
    
    @abstractmethod
    def generate(self, input_text: str, json_mode: bool = False, **kwargs) -> str:

        pass
    
    @abstractmethod
    def get_usage(self) -> ModelUsage:

        pass


class EmbeddingModel:
    
    def __init__(self, model_name: str, device: str = "auto", cache_dir: Optional[str] = None):

        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_dir = cache_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(self.device)
        
    def _get_device(self, device: str) -> str:

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> torch.Tensor:

        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def similarity(self, text1: str, text2: str) -> float:

        embeddings = self.encode([text1, text2])
        emb1, emb2 = embeddings[0], embeddings[1]
        
        similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return similarity.item()


class OpenSourceSQLModel(BaseModel):
    
    def __init__(self, model_name: str, device: str = "auto", cache_dir: Optional[str] = None):

        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_dir = cache_dir
        self.usage_stats = []
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
    def _get_device(self, device: str) -> str:

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def generate(self, input_text: str, max_new_tokens: int = 512, 
                temperature: float = 0.1, top_p: float = 0.9, json_mode: bool = False, **kwargs) -> str:

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        output_length = len(generated_ids)
        usage = ModelUsage(
            input_tokens=input_length,
            output_tokens=output_length,
            model_name=self.model_name
        )
        self.usage_stats.append(usage)
        
        return generated_text.strip()
    
    def get_usage(self) -> ModelUsage:
        return self.usage_stats[-1] if self.usage_stats else ModelUsage(0, 0, model_name=self.model_name)


class CommercialAPIModel(BaseModel):
    
    def __init__(self, model_config: Dict[str, Any]):

        self.model_config = model_config
        self.model_name = model_config['model']
        self.api_type = self._detect_api_type()
        self.usage_stats = []
        
        self.tokenizer = None
        if 'gpt' in self.model_name.lower():
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self._init_api_client()
    
    def _detect_api_type(self) -> str:

        model_name_lower = self.model_name.lower()
        if 'gpt' in model_name_lower or 'openai' in model_name_lower:
            return 'openai'
        elif 'gemini' in model_name_lower or 'google' in model_name_lower:
            return 'google'
        else:
            raise ValueError(f"Unknown API type for model: {self.model_name}")
    
    def _init_api_client(self):
        if self.api_type == 'openai':
            try:
                import openai
                api_key = self.model_config.get('api_key') or os.getenv(self.model_config.get('api_key_env', 'OPENAI_API_KEY'))
                if not api_key:
                    raise ValueError(f"API key not found in config or environment. Set 'api_key' in config or {self.model_config.get('api_key_env', 'OPENAI_API_KEY')} environment variable")
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required for GPT models. Install with: pip install openai")
        
        elif self.api_type == 'google':
            try:
                import google.generativeai as genai
                api_key = self.model_config.get('api_key') or os.getenv(self.model_config.get('api_key_env', 'GEMINI_API_KEY'))
                if not api_key:
                    raise ValueError(f"API key not found in config or environment. Set 'api_key' in config or {self.model_config.get('api_key_env', 'GEMINI_API_KEY')} environment variable")
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError("google-generativeai package required for Gemini models. Install with: pip install google-generativeai")
    
    def generate(self, input_text: str, json_mode: bool = False, **kwargs) -> str:

        if self.api_type == 'openai':
            return self._generate_openai(input_text, json_mode=json_mode, **kwargs)
        elif self.api_type == 'google':
            return self._generate_google(input_text, json_mode=json_mode, **kwargs)
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
    
    def _generate_openai(self, input_text: str, json_mode: bool = False, **kwargs) -> str:

        try:
            request_params = {
                'model': self.model_name,
                'messages': [{"role": "user", "content": input_text}],
                'max_tokens': kwargs.get('max_tokens', self.model_config.get('max_tokens', 1000)),
                'temperature': kwargs.get('temperature', self.model_config.get('temperature', 0.1)),
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature', 'json_mode']}
            }
            
            if json_mode and 'gpt-4' in self.model_name.lower():
                request_params['response_format'] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**request_params)
            
            generated_text = response.choices[0].message.content
            
            cost = self._calculate_openai_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            usage = ModelUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model_name=self.model_name,
                cost=cost
            )
            self.usage_stats.append(usage)
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_api_cost(
                    'OpenAI',
                    self.model_name,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cost,
                    'SQL Generation'
                )
            
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def _generate_google(self, input_text: str, json_mode: bool = False, **kwargs) -> str:

        try:

            generation_config = {
                'max_output_tokens': kwargs.get('max_tokens', self.model_config.get('max_tokens', 1000)),
                'temperature': kwargs.get('temperature', self.model_config.get('temperature', 0.1)),
            }
            
            
            response = self.client.generate_content(
                input_text,
                generation_config=generation_config
            )
            
            generated_text = response.text
            
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(generated_text)
            
            cost = self._calculate_google_cost(input_tokens, output_tokens)
            usage = ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_name=self.model_name,
                cost=cost
            )
            self.usage_stats.append(usage)
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_api_cost(
                    'Google Gemini',
                    self.model_name,
                    input_tokens,
                    output_tokens,
                    cost,
                    'SQL Generation'
                )
            
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Google Gemini API error: {str(e)}")
    
    def _estimate_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4
    
    def _calculate_openai_cost(self, input_tokens: int, output_tokens: int) -> float:

        if 'gpt-4o' in self.model_name.lower():
            input_cost = input_tokens * 0.000005  # $5 per 1M tokens
            output_cost = output_tokens * 0.000015  # $15 per 1M tokens
            return input_cost + output_cost
        return 0.0
    
    def _calculate_google_cost(self, input_tokens: int, output_tokens: int) -> float:

        if 'gemini' in self.model_name.lower():
            input_cost = input_tokens * 0.000001  # Estimated
            output_cost = output_tokens * 0.000002  # Estimated
            return input_cost + output_cost
        return 0.0
    
    def get_usage(self) -> ModelUsage:
        return self.usage_stats[-1] if self.usage_stats else ModelUsage(0, 0, model_name=self.model_name)


class ModelManager:
    
    def __init__(self, config_manager):

        self.config = config_manager
        self.models = {}
        self.usage_tracker = []
        self.logger = None
        
        self.local_models_path = config_manager.get('models.local_models_path')
        if self.local_models_path:
            os.makedirs(self.local_models_path, exist_ok=True)
    
    def set_logger(self, logger):
        self.logger = logger
    
    def load_embedding_model(self) -> EmbeddingModel:

        if 'embedding' not in self.models:
            embedding_config = self.config.get_model_config('embedding')
            model_name = embedding_config['model_name']
            
            if self.logger:
                self.logger.log_model_loading(model_name, "embedding")
            
            self.models['embedding'] = EmbeddingModel(
                model_name=model_name,
                device=embedding_config.get('device', 'auto'),
                cache_dir=self.local_models_path
            )
            
            if self.logger:
                self.logger.log_model_loaded(model_name, "embedding")
        
        return self.models['embedding']
    
    def load_sql_generation_model(self, model_name: str, local_path: Optional[str] = None) -> OpenSourceSQLModel:

        return self.load_local_model(model_name, "SQL generation", local_path)
    
    def load_commercial_model(self, model_name: str, model_purpose: str = "unspecified") -> CommercialAPIModel:

        enable_reuse = self.config.get('models.enable_model_reuse', True)
        
        if enable_reuse:
            cache_key = f"commercial_{model_name}"
        else:
            cache_key = f"commercial_{model_purpose.replace(' ', '_').lower()}_{model_name}"
        
        if cache_key not in self.models:
            commercial_config = self.config.get_model_config('commercial')
            
            if model_name not in commercial_config:
                raise ValueError(f"Commercial model {model_name} not configured")
            
            if self.logger:
                self.logger.log_model_loading(model_name, model_purpose)
            
            model_instance = CommercialAPIModel(commercial_config[model_name])
            if self.logger:
                model_instance.logger = self.logger
                self.logger.log_model_loaded(model_name, model_purpose)
            
            self.models[cache_key] = model_instance
            
            self.models[cache_key]._purposes = [model_purpose]
        else:
            if enable_reuse:
                if not hasattr(self.models[cache_key], '_purposes'):
                    self.models[cache_key]._purposes = []
                
                if model_purpose not in self.models[cache_key]._purposes:
                    self.models[cache_key]._purposes.append(model_purpose)
                    
                    if self.logger:
                        existing_purposes = ", ".join(self.models[cache_key]._purposes[:-1])
                        self.logger.info(f"♻️  Reusing commercial model {model_name} for {model_purpose} (previously loaded for: {existing_purposes})")
        
        return self.models[cache_key]
    
    def load_local_model(self, model_name: str, model_purpose: str, local_path: Optional[str] = None) -> OpenSourceSQLModel:

        enable_reuse = self.config.get('models.enable_model_reuse', True)
        
        if enable_reuse:
            cache_key = f"local_{model_name}_{local_path or 'default'}"
        else:
            cache_key = f"local_{model_purpose.replace(' ', '_').lower()}_{model_name}_{local_path or 'default'}"
        
        if cache_key not in self.models:
            if self.logger:
                self.logger.log_model_loading(model_name, model_purpose)
            
            model_path = local_path if local_path else model_name
            cache_dir = None if local_path else self.local_models_path
            
            self.models[cache_key] = OpenSourceSQLModel(
                model_name=model_path,
                device='auto',
                cache_dir=cache_dir
            )
            
            if self.logger:
                self.logger.log_model_loaded(model_name, model_purpose)
            
            self.models[cache_key]._purposes = [model_purpose]
        else:
            if enable_reuse:
                if not hasattr(self.models[cache_key], '_purposes'):
                    self.models[cache_key]._purposes = []
                
                if model_purpose not in self.models[cache_key]._purposes:
                    self.models[cache_key]._purposes.append(model_purpose)
                    
                    if self.logger:
                        existing_purposes = ", ".join(self.models[cache_key]._purposes[:-1])
                        self.logger.info(f"♻️  Reusing model {model_name} for {model_purpose} (previously loaded for: {existing_purposes})")
        
        return self.models[cache_key]

    def load_ranking_model(self, model_name: str, model_type: str = "commercial", local_path: Optional[str] = None) -> Union[OpenSourceSQLModel, CommercialAPIModel]:

        if model_type == "commercial":
            return self.load_commercial_model(model_name, "SQL ranking")
        elif model_type == "local":
            return self.load_local_model(model_name, "SQL ranking", local_path)
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'local' or 'commercial'")
    
    def load_refinement_model(self, model_name: str, model_type: str = "commercial", local_path: Optional[str] = None) -> Union[OpenSourceSQLModel, CommercialAPIModel]:

        if model_type == "commercial":
            return self.load_commercial_model(model_name, "schema refinement")
        elif model_type == "local":
            return self.load_local_model(model_name, "schema refinement", local_path)
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'local' or 'commercial'")
    
    def load_sql_fixing_model(self, model_name: str, model_type: str = "commercial", local_path: Optional[str] = None) -> Union[OpenSourceSQLModel, CommercialAPIModel]:

        if model_type == "commercial":
            return self.load_commercial_model(model_name, "SQL fixing")
        elif model_type == "local":
            return self.load_local_model(model_name, "SQL fixing", local_path)
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'local' or 'commercial'")
    
    def get_available_sql_models(self) -> List[str]:

        sql_config = self.config.get_model_config('sql_generation')
        return [model['name'] for model in sql_config]
    
    def get_available_commercial_models(self) -> List[str]:

        commercial_config = self.config.get_model_config('commercial')
        return list(commercial_config.keys())
    
    def clear_model_cache(self, model_type: Optional[str] = None) -> None:

        if model_type:
            keys_to_remove = [k for k in self.models.keys() if k.startswith(model_type)]
            for key in keys_to_remove:
                del self.models[key]
        else:
            self.models.clear()
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_reuse_summary(self) -> Dict[str, Any]:

        reuse_summary = {
            'total_models_loaded': len(self.models),
            'models_with_multiple_purposes': 0,
            'total_purposes': 0,
            'reuse_efficiency': 0.0,
            'model_details': {}
        }
        
        for model_key, model in self.models.items():
            if hasattr(model, '_purposes') and model._purposes:
                purposes = model._purposes
                reuse_summary['total_purposes'] += len(purposes)
                
                if len(purposes) > 1:
                    reuse_summary['models_with_multiple_purposes'] += 1
                
                if model_key.startswith('local_'):
                    model_name = model_key.replace('local_', '').replace('_default', '')
                elif model_key.startswith('commercial_'):
                    model_name = model_key.replace('commercial_', '')
                else:
                    model_name = model_key
                
                reuse_summary['model_details'][model_name] = {
                    'purposes': purposes,
                    'purpose_count': len(purposes),
                    'is_reused': len(purposes) > 1
                }
        
        if reuse_summary['total_purposes'] > 0:
            reuse_summary['reuse_efficiency'] = (
                (reuse_summary['total_purposes'] - len(self.models)) / 
                reuse_summary['total_purposes'] * 100
            )
        
        return reuse_summary

    def get_model_usage_summary(self) -> Dict[str, Any]:

        summary = {
            'total_models_loaded': len(self.models),
            'model_types': {},
            'usage_by_model': {},
            'total_cost': 0.0,
            'total_tokens': {'input': 0, 'output': 0}
        }
        
        for model_key, model in self.models.items():
            if hasattr(model, 'usage_stats') and model.usage_stats:
                model_usage = []
                total_input = 0
                total_output = 0
                total_cost = 0.0
                
                for usage in model.usage_stats:
                    total_input += usage.input_tokens
                    total_output += usage.output_tokens
                    if usage.cost:
                        total_cost += usage.cost
                
                summary['usage_by_model'][model_key] = {
                    'total_calls': len(model.usage_stats),
                    'input_tokens': total_input,
                    'output_tokens': total_output,
                    'cost': total_cost
                }
                
                summary['total_tokens']['input'] += total_input
                summary['total_tokens']['output'] += total_output
                summary['total_cost'] += total_cost
        
        for model_key in self.models.keys():
            if model_key.startswith('sql_gen_'):
                summary['model_types']['sql_generation'] = summary['model_types'].get('sql_generation', 0) + 1
            elif model_key.startswith('commercial_'):
                summary['model_types']['commercial'] = summary['model_types'].get('commercial', 0) + 1
            elif model_key == 'embedding':
                summary['model_types']['embedding'] = 1
        
        return summary
    
    def export_usage_report(self, output_path: str) -> None:

        import json
        from datetime import datetime
        
        detailed_report = {
            'report_generated': datetime.now().isoformat(),
            'summary': self.get_model_usage_summary(),
            'detailed_usage': {}
        }
        
        for model_key, model in self.models.items():
            if hasattr(model, 'usage_stats') and model.usage_stats:
                detailed_report['detailed_usage'][model_key] = [
                    {
                        'input_tokens': usage.input_tokens,
                        'output_tokens': usage.output_tokens,
                        'cost': usage.cost,
                        'timestamp': usage.timestamp
                    }
                    for usage in model.usage_stats
                ]
        
        with open(output_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
    
    def cleanup(self) -> None:
        
        self.clear_model_cache()
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
