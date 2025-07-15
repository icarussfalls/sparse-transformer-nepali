import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from utils import get_model
from train import get_ds
from config import get_config, auto_configure_paths

def analyze_sparsity(model_type='adaptive_sparse', checkpoint_epoch=14):
    """Analyzes and visualizes attention sparsity patterns"""
    
    # Setup configuration
    config = get_config()
    if model_type == 'adaptive_sparse':
        config['use_sparse'] = False
        config['use_adaptive_sparse'] = True
        config['attn_type'] = "sparsemax"  # or "entmax15"
    
    # Auto-configure paths and load model
    config = auto_configure_paths(config)
    config['preload'] = str(checkpoint_epoch)
    
    # Load data and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    # Create output directory
    output_dir = Path(f"sparsity_analysis/{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample sentences for analysis
    sample_texts = [
        "How are you doing today?",
        "Nepal is a beautiful country.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    # Analysis metrics
    sparsity_stats = {
        'encoder': [],
        'decoder_self': [],
        'decoder_cross': []
    }
    
    def compute_sparsity(attention_weights):
        """Compute sparsity metrics for attention weights"""
        # Consider weights < 1e-10 as zero (numerical threshold)
        zero_mask = (attention_weights.abs() < 1e-10)
        sparsity = zero_mask.float().mean().item()
        active_heads = (~zero_mask).any(dim=-1).any(dim=-1).float().mean().item()
        return {
            'sparsity': sparsity,
            'active_heads': active_heads,
            'max_value': attention_weights.max().item(),
            'entropy': -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1).mean().item()
        }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i, text in enumerate(sample_texts):
            # Prepare input
            encoder_input = tokenizer_src.encode(text).ids
            encoder_input = torch.tensor([encoder_input], device=device)
            encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).to(device)
            
            # Get attention weights
            translation, attention_maps = model.translate_with_attention(
                encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt
            )
            
            # Analyze encoder self-attention
            for layer_idx, layer_attn in enumerate(attention_maps['encoder']):
                stats = compute_sparsity(layer_attn)
                sparsity_stats['encoder'].append({
                    'layer': layer_idx,
                    'sample': i,
                    **stats
                })
                
                # Visualize sparsity pattern
                plt.figure(figsize=(15, 5))
                
                # Plot 1: Attention heatmap
                plt.subplot(1, 2, 1)
                sns.heatmap(layer_attn[0, 0].cpu().numpy(), cmap='viridis')
                plt.title(f'Layer {layer_idx} Attention Pattern')
                
                # Plot 2: Sparsity histogram
                plt.subplot(1, 2, 2)
                plt.hist(layer_attn.cpu().numpy().flatten(), bins=50)
                plt.title(f'Attention Weight Distribution\nSparsity: {stats["sparsity"]:.2%}')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'sample{i}_layer{layer_idx}_sparsity.png')
                plt.close()
            
            # Save sparsity statistics
            with open(output_dir / f'sample{i}_stats.txt', 'w') as f:
                f.write(f"Input text: {text}\n\n")
                f.write("Encoder Self-Attention Statistics:\n")
                for stat in sparsity_stats['encoder']:
                    if stat['sample'] == i:
                        f.write(f"Layer {stat['layer']}:\n")
                        f.write(f"  Sparsity: {stat['sparsity']:.2%}\n")
                        f.write(f"  Active heads: {stat['active_heads']:.2%}\n")
                        f.write(f"  Max attention value: {stat['max_value']:.4f}\n")
                        f.write(f"  Attention entropy: {stat['entropy']:.4f}\n\n")
    
            # Analyze decoder cross-attention
            if 'decoder_cross' in attention_maps:
                for step_idx, step_attns in enumerate(attention_maps['decoder_cross']):
                    for layer_idx, layer_attn in enumerate(step_attns):
                        stats = compute_sparsity(layer_attn)
                        sparsity_stats['decoder_cross'].append({
                            'layer': layer_idx,
                            'step': step_idx,
                            'sample': i,
                            **stats
                        })
                        
                        # Visualize decoder attention
                        plt.figure(figsize=(15, 5))
                        
                        # Plot 1: Cross-attention heatmap
                        plt.subplot(1, 2, 1)
                        sns.heatmap(layer_attn[0, 0].cpu().numpy(), cmap='viridis')
                        plt.title(f'Decoder Layer {layer_idx} Step {step_idx} Cross-Attention')
                        
                        # Plot 2: Sparsity histogram
                        plt.subplot(1, 2, 2)
                        plt.hist(layer_attn.cpu().numpy().flatten(), bins=50)
                        plt.title(f'Cross-Attention Distribution\nSparsity: {stats["sparsity"]:.2%}')
                        
                        plt.tight_layout()
                        plt.savefig(output_dir / f'sample{i}_decoder_layer{layer_idx}_step{step_idx}_cross_attention.png')
                        plt.close()

    # Plot overall sparsity patterns
    plt.figure(figsize=(10, 6))
    sparsities = [s['sparsity'] for s in sparsity_stats['encoder']]
    layers = [s['layer'] for s in sparsity_stats['encoder']]
    sns.boxplot(x=layers, y=sparsities)
    plt.xlabel('Layer')
    plt.ylabel('Sparsity Ratio')
    plt.title('Attention Sparsity Across Layers')
    plt.savefig(output_dir / 'overall_sparsity_pattern.png')
    plt.close()

if __name__ == "__main__":
    analyze_sparsity()