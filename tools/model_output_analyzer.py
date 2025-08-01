import sys
import json
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config_manager import ConfigManager
from src.core.logger import PipelineLogger


class ModelOutputAnalyzer:
    
    def __init__(self, json_log_file: str = None, config_path: str = None):

        self.json_log_file = json_log_file
        
        if json_log_file and Path(json_log_file).exists():
            with open(json_log_file, 'r') as f:
                self.logs = json.load(f)
        elif config_path:
            config = ConfigManager(config_path)
            self.logger = PipelineLogger(config)
            self.logs = self.logger.json_logs
        else:
            self.logs = []
    
    def get_model_outputs_for_question(self, question_id: int) -> List[Dict[str, Any]]:

        model_outputs = []
        for log_entry in self.logs:
            extra_data = log_entry.get('extra_data', {})
            if (extra_data.get('model_output_capture') and 
                extra_data.get('question_id') == question_id):
                model_outputs.append(extra_data)
        return model_outputs
    
    def get_model_outputs_by_phase(self, question_id: int, phase: str) -> List[Dict[str, Any]]:

        model_outputs = []
        for log_entry in self.logs:
            extra_data = log_entry.get('extra_data', {})
            if (extra_data.get('model_output_capture') and 
                extra_data.get('question_id') == question_id and
                extra_data.get('phase') == phase):
                model_outputs.append(extra_data)
        return model_outputs
    
    def analyze_question_model_usage(self, question_id: int) -> Dict[str, Any]:

        outputs = self.get_model_outputs_for_question(question_id)
        
        if not outputs:
            return {
                'question_id': question_id,
                'error': 'No model outputs found for this question'
            }
        
        by_phase = {}
        by_purpose = {}
        by_model = {}
        
        for output in outputs:
            phase = output.get('phase', 'unknown')
            purpose = output.get('model_purpose', 'unknown')
            model_name = output.get('model_name', 'unknown')
            
            if phase not in by_phase:
                by_phase[phase] = []
            by_phase[phase].append(output)
            
            if purpose not in by_purpose:
                by_purpose[purpose] = []
            by_purpose[purpose].append(output)
            
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(output)
        
        return {
            'question_id': question_id,
            'total_model_calls': len(outputs),
            'phases_involved': list(by_phase.keys()),
            'purposes_used': list(by_purpose.keys()),
            'models_used': list(by_model.keys()),
            'detailed_breakdown': {
                'by_phase': {phase: len(calls) for phase, calls in by_phase.items()},
                'by_purpose': {purpose: len(calls) for purpose, calls in by_purpose.items()},
                'by_model': {model: len(calls) for model, calls in by_model.items()}
            },
            'model_outputs': outputs
        }
    
    def extract_model_reasoning(self, question_id: int, purpose: str = None) -> List[Dict[str, Any]]:

        if purpose:
            outputs = self.get_model_outputs_by_purpose(question_id, purpose)
        else:
            outputs = self.get_model_outputs_for_question(question_id)
        
        reasoning_data = []
        for output in outputs:
            reasoning_data.append({
                'phase': output.get('phase'),
                'purpose': output.get('model_purpose'),
                'model_name': output.get('model_name'),
                'prompt': output.get('prompt'),
                'raw_output': output.get('raw_output'),
                'parsed_output': output.get('parsed_output'),
                'parsing_success': output.get('parsing_success', True),
                'output_length': output.get('output_length', 0)
            })
        
        return reasoning_data
    
    def get_model_outputs_by_purpose(self, question_id: int, model_purpose: str) -> List[Dict[str, Any]]:
 
        model_outputs = []
        for log_entry in self.logs:
            extra_data = log_entry.get('extra_data', {})
            if (extra_data.get('model_output_capture') and 
                extra_data.get('question_id') == question_id and
                extra_data.get('model_purpose') == model_purpose):
                model_outputs.append(extra_data)
        return model_outputs
    
    def save_model_outputs_for_question(self, question_id: int, output_file: str):

        analysis = self.analyze_question_model_usage(question_id)
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Model outputs for question {question_id} saved to {output_file}")
    
    def print_question_summary(self, question_id: int):

        analysis = self.analyze_question_model_usage(question_id)
        
        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
            return
        
        print(f"\n📊 MODEL OUTPUT SUMMARY FOR QUESTION {question_id}")
        print("=" * 60)
        print(f"Total model calls: {analysis['total_model_calls']}")
        print(f"Phases involved: {', '.join(analysis['phases_involved'])}")
        print(f"Models used: {', '.join(analysis['models_used'])}")
        
        print(f"\n📋 BREAKDOWN BY PHASE:")
        for phase, count in analysis['detailed_breakdown']['by_phase'].items():
            print(f"   {phase}: {count} calls")
        
        print(f"\n🎯 BREAKDOWN BY PURPOSE:")
        for purpose, count in analysis['detailed_breakdown']['by_purpose'].items():
            print(f"   {purpose}: {count} calls")
        
        print(f"\n🤖 BREAKDOWN BY MODEL:")
        for model, count in analysis['detailed_breakdown']['by_model'].items():
            print(f"   {model}: {count} calls")
        
        print("=" * 60)


def main():

    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze model outputs from pipeline execution')
    parser.add_argument('--log-file', help='Path to JSON log file from pipeline execution')
    parser.add_argument('--config', help='Path to pipeline configuration file')
    parser.add_argument('--question-id', type=int, help='Question ID to analyze')
    parser.add_argument('--save-outputs', help='File to save model outputs to')
    parser.add_argument('--purpose', help='Filter by model purpose (schema_refinement, sql_generation, etc.)')
    parser.add_argument('--phase', help='Filter by phase (phase1, phase2, phase3)')
    
    args = parser.parse_args()
    
    if not args.log_file and not args.config:
        print("❌ Error: Must provide either --log-file or --config")
        sys.exit(1)
    
    analyzer = ModelOutputAnalyzer(args.log_file, args.config)
    
    if args.question_id is not None:
        if args.save_outputs:
            analyzer.save_model_outputs_for_question(args.question_id, args.save_outputs)
        else:
            analyzer.print_question_summary(args.question_id)
            
            if args.purpose or args.phase:
                print(f"\n🔍 DETAILED MODEL OUTPUTS:")
                if args.purpose:
                    outputs = analyzer.get_model_outputs_by_purpose(args.question_id, args.purpose)
                elif args.phase:
                    outputs = analyzer.get_model_outputs_by_phase(args.question_id, args.phase)
                
                for i, output in enumerate(outputs, 1):
                    print(f"\n--- OUTPUT {i} ---")
                    print(f"Model: {output.get('model_name')}")
                    print(f"Purpose: {output.get('model_purpose')}")
                    print(f"Phase: {output.get('phase')}")
                    print(f"Parsing Success: {output.get('parsing_success', True)}")
                    
                    prompt = output.get('prompt', '')
                    if len(prompt) > 200:
                        prompt = prompt[:200] + "..."
                    print(f"Prompt: {prompt}")
                    
                    raw_output = output.get('raw_output', '')
                    if len(raw_output) > 300:
                        raw_output = raw_output[:300] + "..."
                    print(f"Output: {raw_output}")
    else:
        print("❌ Error: Must specify --question-id to analyze")
        sys.exit(1)


if __name__ == "__main__":
    main()