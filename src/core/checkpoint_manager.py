import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

class CheckpointManager:
    
    def __init__(self, config_manager):

        self.config = config_manager
        self.checkpoints_dir = Path(config_manager.get('data.checkpoints_path', 'data/checkpoints'))
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.main_checkpoint_file = "main_checkpoint.json"
        self.candidate_pools_checkpoint = "candidate_pools_checkpoint.json"
        self.session_checkpoint = "session_checkpoint.json"
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], 
                       checkpoint_name: Optional[str] = None) -> str:
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{timestamp}"
        
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}.json"
        
        checkpoint_data['checkpoint_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_name': checkpoint_name,
            'config_hash': self._get_config_hash()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        latest_path = self.checkpoints_dir / self.main_checkpoint_file
        shutil.copy2(checkpoint_path, latest_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:

        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}.json"
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_name}: {str(e)}")
            return None
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:

        latest_path = self.checkpoints_dir / self.main_checkpoint_file
        if not latest_path.exists():
            return None
        
        try:
            with open(latest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading latest checkpoint: {str(e)}")
            return None
    
    def save_candidate_pools_checkpoint(self, pools_data: Dict[str, Any]) -> str:

        checkpoint_path = self.checkpoints_dir / self.candidate_pools_checkpoint
        
        pools_data['checkpoint_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_type': 'candidate_pools',
            'config_hash': self._get_config_hash()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(pools_data, f, indent=2)
        
        return str(checkpoint_path)
    
    def load_candidate_pools_checkpoint(self) -> Optional[Dict[str, Any]]:

        checkpoint_path = self.checkpoints_dir / self.candidate_pools_checkpoint
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading candidate pools checkpoint: {str(e)}")
            return None
    
    def save_session_checkpoint(self, session_data: Dict[str, Any]) -> str:

        checkpoint_path = self.checkpoints_dir / self.session_checkpoint
        
        session_data['session_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_type': 'session',
            'session_id': session_data.get('session_id', 'unknown')
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return str(checkpoint_path)
    
    def load_session_checkpoint(self) -> Optional[Dict[str, Any]]:

        checkpoint_path = self.checkpoints_dir / self.session_checkpoint
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session checkpoint: {str(e)}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:

        checkpoints = []
        
        for checkpoint_file in self.checkpoints_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get('checkpoint_metadata', {})
                checkpoints.append({
                    'name': checkpoint_file.stem,
                    'file': str(checkpoint_file),
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'type': metadata.get('checkpoint_type', 'general'),
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024)
                })
            except Exception:
                continue
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:

        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}.json"
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                return True
            except Exception as e:
                print(f"Error deleting checkpoint {checkpoint_name}: {str(e)}")
                return False
        return False
    
    def clean_old_checkpoints(self, keep_count: int = 5) -> int:

        checkpoints = self.list_checkpoints()
        deleted_count = 0
        
        special_checkpoints = {
            self.main_checkpoint_file.replace('.json', ''),
            self.candidate_pools_checkpoint.replace('.json', ''),
            self.session_checkpoint.replace('.json', '')
        }
        
        regular_checkpoints = [
            cp for cp in checkpoints 
            if cp['name'] not in special_checkpoints
        ]
        
        if len(regular_checkpoints) > keep_count:
            checkpoints_to_delete = regular_checkpoints[keep_count:]
            for checkpoint in checkpoints_to_delete:
                if self.delete_checkpoint(checkpoint['name']):
                    deleted_count += 1
        
        return deleted_count
    
    def get_checkpoint_status(self) -> Dict[str, Any]:

        checkpoints = self.list_checkpoints()
        latest = self.load_latest_checkpoint()
        
        total_size_mb = sum(cp['size_mb'] for cp in checkpoints)
        
        return {
            'checkpoints_directory': str(self.checkpoints_dir),
            'total_checkpoints': len(checkpoints),
            'total_size_mb': round(total_size_mb, 2),
            'latest_checkpoint': latest.get('checkpoint_metadata', {}) if latest else None,
            'has_candidate_pools_checkpoint': (self.checkpoints_dir / self.candidate_pools_checkpoint).exists(),
            'has_session_checkpoint': (self.checkpoints_dir / self.session_checkpoint).exists()
        }
    
    def create_phase2_checkpoint(self, question_idx: int, completed_configs: List[str], 
                                current_config: Optional[str] = None, 
                                candidate_pools: Optional[Dict] = None) -> str:

        phase2_data = {
            'phase': 'phase2_sql_generation',
            'question_index': question_idx,
            'completed_configurations': completed_configs,
            'current_configuration': current_config,
            'candidate_pools': candidate_pools or {},
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_name = f"phase2_q{question_idx}_{datetime.now().strftime('%H%M%S')}"
        return self.save_checkpoint(phase2_data, checkpoint_name)
    
    def load_phase2_checkpoint(self) -> Optional[Dict[str, Any]]:

        checkpoints = self.list_checkpoints()
        phase2_checkpoints = [
            cp for cp in checkpoints 
            if cp['name'].startswith('phase2_')
        ]
        
        if not phase2_checkpoints:
            return None
        
        latest_phase2 = phase2_checkpoints[0]
        return self.load_checkpoint(latest_phase2['name'])
    
    def _get_config_hash(self) -> str:

        import hashlib
        
        config_str = json.dumps(self.config.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def validate_checkpoint_compatibility(self, checkpoint_data: Dict[str, Any]) -> bool:

        if 'checkpoint_metadata' not in checkpoint_data:
            return False
        
        checkpoint_config_hash = checkpoint_data['checkpoint_metadata'].get('config_hash')
        current_config_hash = self._get_config_hash()
        
        if checkpoint_config_hash != current_config_hash:
            print("Warning: Checkpoint was created with different configuration")
            print(f"Checkpoint config hash: {checkpoint_config_hash}")
            print(f"Current config hash: {current_config_hash}")
        
        return True
    
    def export_checkpoint_summary(self, output_path: str) -> None:

        summary = {
            'checkpoint_status': self.get_checkpoint_status(),
            'checkpoint_list': self.list_checkpoints(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
