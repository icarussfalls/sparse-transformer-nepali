"""
Simple Architecture Comparison Script for Kaggle
Compares Vanilla, Sparse, and Adaptive Sparse Transformers
"""

import torch
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Import your existing modules
from train import get_ds, get_model, train_model
from config import get_config

# Try to import get_architecture_config, create fallback if not available
try:
    from config import get_architecture_config
except ImportError:
    print("⚠️  get_architecture_config not found, using fallback")
    
    def get_architecture_config(arch_type):
        """Fallback architecture config"""
        base_config = get_config()
        
        arch_configs = {
            'vanilla': {
                'use_sparse': False,
                'use_adaptive_sparse': False,
                'model_folder': 'weights_vanilla',
                'experiment_name': 'runs/vanilla',
                'description': 'Standard Transformer'
            },
            'sparse': {
                'use_sparse': True,
                'use_adaptive_sparse': False,
                'sparse_block_size': 64,
                'sparse_stride': 32,
                'model_folder': 'weights_sparse',
                'experiment_name': 'runs/sparse',
                'description': 'Fixed Sparse Attention'
            },
            'adaptive_sparse': {
                'use_adaptive_sparse': True,
                'use_sparse': False,
                'attn_type': 'adaptive',
                'model_folder': 'weights_adaptive_sparse',
                'experiment_name': 'runs/adaptive_sparse',
                'description': 'Adaptive Sparse Attention'
            }
        }
        
        if arch_type not in arch_configs:
            raise ValueError(f"Unknown architecture: {arch_type}")
        
        return {**base_config, **arch_configs[arch_type]}

class SimpleComparison:
    def __init__(self):
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
    def run_experiment(self, arch_name):
        """Run single experiment with architecture-specific config"""
        print(f"\n🚀 Running {arch_name.upper()}...")
        
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        try:
            # Get architecture-specific config
            config = get_architecture_config(arch_name)
            
            # Set device and GPU configuration
            config['device'] = self.device
            config['gpu'] = 0 if self.device == 'cuda' else None
            
            # Reduce epochs for quick comparison
            config['num_epochs'] = 5
            config['batch_size'] = 16 if self.device == 'cuda' else 8
            config['val_batch_size'] = 8 if self.device == 'cuda' else 4
            
            # Disable preloading for comparison
            config['preload'] = False
            
            print(f"📁 Model folder: {config['model_folder']}")
            print(f"📊 Description: {config['description']}")
            print(f"🔧 Device: {config['device']}")
            print(f"🎯 GPU: {config['gpu']}")
            
            # Get data
            train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
            
            # Get model
            model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"🔢 Total params: {total_params:,}")
            print(f"🔢 Trainable params: {trainable_params:,}")
            
            # Train
            train_model(config, model, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt)
            
            # Record results
            end_time = time.time()
            peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            
            self.results[arch_name] = {
                'training_time': (end_time - start_time) / 60,
                'parameters': total_params,
                'trainable_parameters': trainable_params,
                'peak_memory': peak_memory,
                'device': self.device,
                'model_folder': config['model_folder'],
                'experiment_name': config['experiment_name'],
                'description': config['description'],
                'status': 'completed'
            }
            
            print(f"✅ {arch_name} done! Time: {self.results[arch_name]['training_time']:.1f}min")
            print(f"💾 Saved to: {config['model_folder']}")
            
        except Exception as e:
            print(f"❌ {arch_name} failed: {e}")
            import traceback
            traceback.print_exc()  # Print full error trace for debugging
            self.results[arch_name] = {'status': 'failed', 'error': str(e)}
            
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_all(self):
        """Run all experiments"""
        architectures = ['vanilla', 'sparse', 'adaptive_sparse']
        
        print("🔬 Starting Architecture Comparison")
        print(f"🔧 Device: {self.device}")
        print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 50)
        
        for arch_name in architectures:
            self.run_experiment(arch_name)
        
        self.print_results()
        self.plot_results()
        
    def print_results(self):
        """Print comparison table"""
        print("\n" + "=" * 90)
        print("📊 ARCHITECTURE COMPARISON RESULTS")
        print("=" * 90)
        print(f"{'Architecture':<15} {'Status':<10} {'Time(min)':<10} {'Memory(GB)':<10} {'Params(M)':<10} {'Description':<25}")
        print("-" * 90)
        
        for arch, result in self.results.items():
            if result['status'] == 'completed':
                params_m = result['parameters'] / 1e6
                desc = result.get('description', 'N/A')[:24]
                print(f"{arch:<15} {'✅ OK':<10} {result['training_time']:<10.1f} {result['peak_memory']:<10.1f} {params_m:<10.1f} {desc:<25}")
            else:
                print(f"{arch:<15} {'❌ FAIL':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<25}")
        
        print("-" * 90)
        
        # Efficiency comparison
        completed = {k: v for k, v in self.results.items() if v['status'] == 'completed'}
        if len(completed) > 1 and 'vanilla' in completed:
            print("\n⚡ EFFICIENCY COMPARISON (vs Vanilla)")
            print("-" * 50)
            vanilla_time = completed['vanilla']['training_time']
            vanilla_memory = completed['vanilla']['peak_memory']
            
            for arch, result in completed.items():
                if arch != 'vanilla':
                    time_ratio = vanilla_time / result['training_time'] if result['training_time'] > 0 else 1
                    memory_ratio = vanilla_memory / result['peak_memory'] if result['peak_memory'] > 0 else 1
                    
                    print(f"{arch.upper()}:")
                    print(f"  Speed: {time_ratio:.2f}x {'faster' if time_ratio > 1 else 'slower'}")
                    print(f"  Memory: {memory_ratio:.2f}x {'less' if memory_ratio > 1 else 'more'}")
                    print()
        
    def plot_results(self):
        """Simple bar plots"""
        completed = {k: v for k, v in self.results.items() if v['status'] == 'completed'}
        
        if len(completed) < 2:
            print("❌ Not enough results to plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Transformer Architecture Comparison', fontsize=16)
        
        # Training time
        times = [completed[arch]['training_time'] for arch in completed]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(completed)]
        ax1.bar(completed.keys(), times, color=colors)
        ax1.set_title('Training Time')
        ax1.set_ylabel('Minutes')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage
        memory = [completed[arch]['peak_memory'] for arch in completed]
        ax2.bar(completed.keys(), memory, color=colors)
        ax2.set_title('Peak Memory Usage')
        ax2.set_ylabel('GB')
        ax2.tick_params(axis='x', rotation=45)
        
        # Model parameters
        params = [completed[arch]['parameters'] / 1e6 for arch in completed]
        ax3.bar(completed.keys(), params, color=colors)
        ax3.set_title('Model Parameters')
        ax3.set_ylabel('Millions')
        ax3.tick_params(axis='x', rotation=45)
        
        # Efficiency score (Time × Memory)
        efficiency = [times[i] * memory[i] for i in range(len(times))]
        ax4.bar(completed.keys(), efficiency, color=colors)
        ax4.set_title('Efficiency Score (lower = better)')
        ax4.set_ylabel('Time × Memory')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Plots saved to 'comparison_results.png'")
        
    def save_results(self):
        """Save to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'comparison_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"💾 Results saved to '{filename}'")

# Run the comparison
if __name__ == "__main__":
    comparison = SimpleComparison()
    comparison.run_all()
    comparison.save_results()
    print("\n🎉 Comparison complete!")