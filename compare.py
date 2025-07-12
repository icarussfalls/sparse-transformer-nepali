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

class SimpleComparison:
    def __init__(self):
        self.results = {}
        
    def run_experiment(self, arch_name, config):
        """Run single experiment"""
        print(f"\n🚀 Running {arch_name.upper()}...")
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        try:
            # Get data
            train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
            
            # Get model
            model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Train
            train_model(config, model, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt)
            
            # Record results
            end_time = time.time()
            self.results[arch_name] = {
                'training_time': (end_time - start_time) / 60,
                'parameters': total_params,
                'peak_memory': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                'status': 'completed'
            }
            
            print(f"✅ {arch_name} done! Time: {self.results[arch_name]['training_time']:.1f}min")
            
        except Exception as e:
            print(f"❌ {arch_name} failed: {e}")
            self.results[arch_name] = {'status': 'failed', 'error': str(e)}
            
        torch.cuda.empty_cache()
    
    def run_all(self):
        """Run all experiments"""
        base_config = get_config()
        base_config.update({
            'num_epochs': 5,  # Quick comparison
            'batch_size': 16,
            'val_batch_size': 8
        })
        
        # Architectures to test
        experiments = {
            'vanilla': {
                'use_sparse': False,
                'use_adaptive_sparse': False
            },
            'sparse': {
                'use_sparse': True,
                'use_adaptive_sparse': False,
                'sparse_block_size': 64,
                'sparse_stride': 32
            },
            'adaptive': {
                'use_adaptive_sparse': True,
                'use_sparse': False,
                'attn_type': 'adaptive'
            }
        }
        
        print("🔬 Starting Architecture Comparison")
        print("=" * 50)
        
        for arch_name, arch_config in experiments.items():
            config = {**base_config, **arch_config}
            config['experiment_name'] = f'comparison_{arch_name}'
            self.run_experiment(arch_name, config)
        
        self.print_results()
        self.plot_results()
        
    def print_results(self):
        """Print comparison table"""
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"{'Architecture':<12} {'Status':<10} {'Time(min)':<10} {'Memory(GB)':<10}")
        print("-" * 60)
        
        for arch, result in self.results.items():
            if result['status'] == 'completed':
                print(f"{arch:<12} {'✅ OK':<10} {result['training_time']:<10.1f} {result['peak_memory']:<10.1f}")
            else:
                print(f"{arch:<12} {'❌ FAIL':<10} {'N/A':<10} {'N/A':<10}")
        
        print("-" * 60)
        
    def plot_results(self):
        """Simple bar plots"""
        completed = {k: v for k, v in self.results.items() if v['status'] == 'completed'}
        
        if len(completed) < 2:
            print("❌ Not enough results to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training time
        times = [completed[arch]['training_time'] for arch in completed]
        ax1.bar(completed.keys(), times)
        ax1.set_title('Training Time')
        ax1.set_ylabel('Minutes')
        
        # Memory usage
        memory = [completed[arch]['peak_memory'] for arch in completed]
        ax2.bar(completed.keys(), memory)
        ax2.set_title('Peak Memory')
        ax2.set_ylabel('GB')
        
        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Plots saved to 'comparison_results.png'")
        
    def save_results(self):
        """Save to JSON"""
        with open('comparison_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("💾 Results saved to 'comparison_results.json'")

# Run the comparison
if __name__ == "__main__":
    comparison = SimpleComparison()
    comparison.run_all()
    comparison.save_results()
    print("\n🎉 Comparison complete!")