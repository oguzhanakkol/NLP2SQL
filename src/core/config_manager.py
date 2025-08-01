import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigManager:
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._setup_environment()
    
    def _load_config(self) -> Dict[str, Any]:

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self) -> None:

        required_sections = [
            'data', 'logging', 'models', 'phase1_schema_linking',
            'phase2_sql_generation', 'phase3_sql_selection', 'evaluation'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        data_config = self.config['data']
        required_data_fields = ['bird_benchmark_path', 'dev_json_path', 'databases_path']
        for field in required_data_fields:
            if field not in data_config:
                raise ValueError(f"Missing required data configuration: {field}")
        
        models_config = self.config['models']
        if 'embedding' not in models_config:
            raise ValueError("Missing embedding model configuration")
        if 'sql_generation' not in models_config:
            raise ValueError("Missing SQL generation model configuration")
    
    def _setup_environment(self) -> None:

        directories_to_create = [
            self.get('logging.log_directory'),
            self.get('data.checkpoints_path'),
            self.get('data.results_path'),
            self.get('data.candidate_pools_path'),
            self.get('logging.prompts_directory'),
        ]
        
        for directory in directories_to_create:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:

        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:

        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_model_config(self, model_type: str, model_name: Optional[str] = None) -> Dict[str, Any]:

        models_config = self.config.get('models', {})
        
        if model_type == 'embedding':
            return models_config.get('embedding', {})
        elif model_type == 'sql_generation':
            sql_models = models_config.get('sql_generation', {}).get('models', [])
            if model_name:
                for model in sql_models:
                    if model.get('name') == model_name:
                        return model
                raise ValueError(f"SQL generation model not found: {model_name}")
            return sql_models
        elif model_type == 'commercial':
            commercial_models = models_config.get('commercial', {})
            if model_name:
                return commercial_models.get(model_name, {})
            return commercial_models
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_phase_config(self, phase: int) -> Dict[str, Any]:

        phase_key = f"phase{phase}_{'schema_linking' if phase == 1 else 'sql_generation' if phase == 2 else 'sql_selection'}"
        return self.config.get(phase_key, {})
    
    def get_data_paths(self) -> Dict[str, str]:

        data_config = self.config.get('data', {})
        return {
            'bird_benchmark': data_config.get('bird_benchmark_path'),
            'dev_json': data_config.get('dev_json_path'),
            'databases': data_config.get('databases_path'),
            'checkpoints': data_config.get('checkpoints_path'),
            'results': data_config.get('results_path'),
            'candidate_pools': data_config.get('candidate_pools_path')
        }
    
    def get_logging_config(self) -> Dict[str, Any]:

        return self.config.get('logging', {})
    
    def save_config(self, output_path: Optional[str] = None) -> None:

        output_path = output_path or self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:

        def update_nested_dict(d: Dict[str, Any], u: Dict[str, Any]) -> None:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        update_nested_dict(self.config, updates)
    
    def __str__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False, indent=2)
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"
