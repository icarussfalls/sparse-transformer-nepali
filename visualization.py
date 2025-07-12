import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


# function to visualize the alpha values learned
def visualize_alpha_values(model, save_dir='visualizations'):
    """visualize alpha values for adaptive sparse attention"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    alpha_values = []
    layer_names = []
    
    # Fix: Handle module names properly
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and module.alpha is not None:
            alpha_values.append(module.alpha.detach().cpu().numpy())
            # Fix: Extract layer number properly
            if 'encoder' in name and 'layers' in name:
                try:
                    # Extract layer number from name like 'encoder.layers.0.self_attention_block'
                    layer_num = name.split('.layers.')[1].split('.')[0]
                    layer_names.append(f'Encoder Layer {layer_num}')
                except (IndexError, ValueError):
                    layer_names.append(name)  # Fallback to full name
            elif 'decoder' in name and 'layers' in name:
                try:
                    layer_num = name.split('.layers.')[1].split('.')[0]
                    layer_names.append(f'Decoder Layer {layer_num}')
                except (IndexError, ValueError):
                    layer_names.append(name)
            else:
                layer_names.append(name)
    
    if not alpha_values:
        print("No alpha values found in model")
        return
    
    # Plot alpha values
    plt.figure(figsize=(12, 8))
    for i, (alpha, name) in enumerate(zip(alpha_values, layer_names)):
        plt.subplot(2, (len(alpha_values) + 1) // 2, i + 1)
        plt.hist(alpha.flatten(), bins=50, alpha=0.7)
        plt.title(f'{name}\nAlpha Distribution')
        plt.xlabel('Alpha Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/alpha_distributions.png')
    plt.close()

def visualize_attention_patterns(model, tokenizer_src, tokenizer_tgt, sample_text, save_dir='visualizations'):
    """Visualize attention patterns using stored weights"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # Use short sample text
    sample_text = "Hello world"
    tokens = tokenizer_src.encode(sample_text).ids[:8]  # Limit to 8 tokens
    
    # Ensure minimum length
    pad_token = tokenizer_src.token_to_id('[PAD]') or 0
    while len(tokens) < 4:
        tokens.append(pad_token)
    
    input_ids = torch.tensor([tokens]).to(device)
    seq_len = input_ids.size(1)

    # Create masks
    encoder_mask = torch.ones(1, 1, 1, seq_len).to(device)
    decoder_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device)
    decoder_mask = (decoder_mask == 0).float().unsqueeze(0).unsqueeze(0)

    # Forward pass to populate attention weights
    with torch.no_grad():
        try:
            model(input_ids, input_ids, encoder_mask, decoder_mask)
        except Exception as e:
            print(f"Forward pass failed: {e}")
            return

    # Extract attention weights from modules
    layer_idx = 0
    for name, module in model.named_modules():
        if hasattr(module, 'last_attention_weights') and module.last_attention_weights is not None:
            attn_weights = module.last_attention_weights.cpu()
            
            if attn_weights.dim() == 4:  # (B, h, L, L)
                attn_weights = attn_weights[0]  # Remove batch dimension: (h, L, L)
            
            # Plot first 4 heads
            n_heads = min(attn_weights.size(0), 4)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for head_idx in range(n_heads):
                attn_matrix = attn_weights[head_idx].numpy()
                
                sns.heatmap(attn_matrix, 
                           ax=axes[head_idx], 
                           cmap='Blues',
                           cbar=True,
                           square=True,
                           xticklabels=range(seq_len),
                           yticklabels=range(seq_len))
                axes[head_idx].set_title(f'Head {head_idx}')
                axes[head_idx].set_xlabel('Key Position')
                axes[head_idx].set_ylabel('Query Position')
            
            # Hide unused subplots
            for i in range(n_heads, 4):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Layer {layer_idx} Attention Patterns')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/attention_layer_{layer_idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            layer_idx += 1
            if layer_idx >= 2:  # Only first 2 layers
                break
    
    print(f"Attention visualization saved to {save_dir}/")

def log_visualizations(model, tokenizer_src, tokenizer_tgt, global_step, save_dir='visualizations'):
    """this logs the visualizations during training"""
    try:
        if hasattr(model, 'module'):
            model_unwrapped = model.module
        else:
            model_unwrapped = model
        
        step_dir = f'{save_dir}/step_{global_step}'
        
        # Alpha visualization
        print("Generating alpha visualizations...")
        visualize_alpha_values(model_unwrapped, step_dir)
        
        # Attention visualization (now fixed!)
        print("Generating attention visualizations...")
        visualize_attention_patterns(model_unwrapped, tokenizer_src, tokenizer_tgt, 
                                   "Hello world", step_dir)
        
        print(f"Visualizations saved to {step_dir}/")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()