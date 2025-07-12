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
    """this visualizes the attention patterns for a sample input"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # need to tokenize input
    tokens = tokenizer_src.encode(sample_text).ids
    input_ids = torch.tensor([tokens]).to(device)

    # attention mask
    mask = torch.ones((1, 1, 1, len(tokens))).to(device)

    # get attention patterns
    attention_patterns = []
    def hook_fn(module, input, output):
        attention_patterns.append(output.detach().cpu())

    hooks = []
    for name, module in model.named_modules():
        if "self_attention_block" in name:
            hooks.append(module.register_forward_hook(hook_fn))
    
    # forward pass
    with torch.no_grad():
        model(input_ids, input_ids, mask, mask)

    # remove hooks
    for hook in hooks:
        hook.remove()

    # plot attention patterns
    for layer_idx, attn in enumerate(attention_patterns):
        attn = attn[0] # this removes batch dimension
        n_heads = attn.size(0)  # Fixed: was .size[0]

        fig, axes = plt.subplots(2, n_heads//2, figsize=(15,8))  # Fixed: was .subplot
        axes = axes.flat
    
        for head_idx in range(n_heads):
            sns.heatmap(attn[head_idx], ax=axes[head_idx], cmap='viridis')
            axes[head_idx].set_title(f'Head {head_idx}')  # Fixed: was .set_titile

        plt.suptitle(f'Layer {layer_idx} Attention Patterns')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/attention_layer_{layer_idx}.png')
        plt.close()


def log_visualizations(model, tokenizer_src, tokenizer_tgt, global_step, save_dir='visualizations'):
    """this logs the visualizations during training"""
    if hasattr(model, 'module'):
        model_unwrapped = model.module
    else:
        model_unwrapped = model
    
    save_dir = f'{save_dir}/step_{global_step}'
    visualize_alpha_values(model_unwrapped, save_dir)
    sample_text = 'This is a sample input text to visualize attention.'
    visualize_attention_patterns(model_unwrapped, tokenizer_src, tokenizer_tgt, sample_text, save_dir)