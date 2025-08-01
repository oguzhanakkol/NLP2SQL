import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config_manager import ConfigManager
from src.core.logger import PipelineLogger
from src.core.model_manager import ModelManager
from src.core.data_loader import BirdDataLoader
from src.core.statistics_tracker import StatisticsTracker
from src.core.checkpoint_manager import CheckpointManager


def test_config_manager():

    print("Testing Configuration Manager")
    print("=" * 35)
    
    try:
        config = ConfigManager()
        
        log_level = config.get('logging.log_level', 'INFO')
        print(f"✓ Config loaded, log level: {log_level}")
        
        data_paths = config.get_data_paths()
        print(f"✓ Data paths configured: {len(data_paths)} paths")
        
        embedding_config = config.get_model_config('embedding')
        print(f"✓ Embedding model: {embedding_config.get('model_name', 'Not configured')}")
        
        phase1_config = config.get_phase_config(1)
        print(f"✓ Phase 1 config: {len(phase1_config)} settings")
        
        return True
        
    except Exception as e:
        print(f"✗ Config manager test failed: {str(e)}")
        return False


def test_logger():

    print("\nTesting Logger")
    print("=" * 20)
    
    try:
        config = ConfigManager("configs/pipeline_config.yaml")
        logger = PipelineLogger(config)
        
        logger.info("Test log message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        
        print("✓ Basic logging works")
        
        logger.log_model_usage("test_model", "test", 100, 50, 0.01)
        print("✓ Model usage logging works")
        
        summary = logger.get_session_summary()
        print(f"✓ Session summary: {summary['session_id']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Logger test failed: {str(e)}")
        return False


def test_model_manager():

    print("\nTesting Model Manager")
    print("=" * 25)
    
    try:
        config = ConfigManager("configs/pipeline_config.yaml")
        model_manager = ModelManager(config)
        
        sql_models = model_manager.get_available_sql_models()
        commercial_models = model_manager.get_available_commercial_models()
        
        print(f"✓ Available SQL models: {len(sql_models)}")
        print(f"✓ Available commercial models: {len(commercial_models)}")
        
        summary = model_manager.get_model_usage_summary()
        print(f"✓ Usage summary: {summary['total_models_loaded']} models loaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Model manager test failed: {str(e)}")
        return False


def test_statistics_tracker():

    print("\nTesting Statistics Tracker")
    print("=" * 30)
    
    try:
        config = ConfigManager("configs/pipeline_config.yaml")
        tracker = StatisticsTracker(config)
        
        tracker.start_question_processing(1, "test_db", "simple")
        tracker.log_phase_timing(1, "phase1", 1.5)
        tracker.log_token_usage(1, "test_model", 100, 50, 0.01)
        
        print("✓ Question tracking works")
        
        summary = tracker.get_summary()
        print(f"✓ Summary generated: {summary['overview']['total_questions']} questions")
        
        metrics = tracker.get_real_time_metrics()
        print(f"✓ Real-time metrics: {metrics.get('processed_questions', 0)} processed")
        
        return True
        
    except Exception as e:
        print(f"✗ Statistics tracker test failed: {str(e)}")
        return False


def test_checkpoint_manager():

    print("\nTesting Checkpoint Manager")
    print("=" * 30)
    
    try:
        config = ConfigManager("configs/pipeline_config.yaml")
        checkpoint_manager = CheckpointManager(config)
        
        status = checkpoint_manager.get_checkpoint_status()
        print(f"✓ Checkpoint status: {status['total_checkpoints']} checkpoints")
        
        test_data = {"test": "data", "timestamp": "2025-01-01"}
        checkpoint_path = checkpoint_manager.save_checkpoint(test_data, "test_checkpoint")
        print(f"✓ Checkpoint saved: {Path(checkpoint_path).name}")
        
        loaded_data = checkpoint_manager.load_checkpoint("test_checkpoint")
        if loaded_data and loaded_data.get("test") == "data":
            print("✓ Checkpoint loading works")
        else:
            print("✗ Checkpoint loading failed")
            return False
        
        checkpoint_manager.delete_checkpoint("test_checkpoint")
        print("✓ Checkpoint cleanup works")
        
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint manager test failed: {str(e)}")
        return False


def test_data_loader():
    print("\nTesting Data Loader")
    print("=" * 25)
    
    try:
        config = ConfigManager("configs/pipeline_config.yaml")
        data_loader = BirdDataLoader(config)
        
        try:
            stats = data_loader.get_dataset_statistics()
            print(f"✓ Dataset statistics: {stats['total_questions']} questions")
        except Exception:
            print("✓ Data loader initialized (no BIRD data available)")
        
        validation = data_loader.validate_dataset()
        print(f"✓ Dataset validation: {'valid' if validation['valid'] else 'invalid'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {str(e)}")
        return False


def main():
    print("Core Components Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        ("Configuration Manager", test_config_manager),
        ("Logger", test_logger),
        ("Model Manager", test_model_manager),
        ("Statistics Tracker", test_statistics_tracker),
        ("Checkpoint Manager", test_checkpoint_manager),
        ("Data Loader", test_data_loader)
    ]
    
    for test_name, test_func in tests:
        if test_func():
            print(f"✓ {test_name} test PASSED")
        else:
            print(f"✗ {test_name} test FAILED")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All core component tests PASSED!")
        return 0
    else:
        print("❌ Some core component tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
