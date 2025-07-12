import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


# function to visualize the alpha values learned
def visualize_alpha_values(model, save_dir='visualizations'):
    """This will visualize the learned alpha values for each head in the adaptively sparse transformers"""

    # collect alpha values from all attention blocks
    alphas = []
    head_positions = []
    layer_number = []

    for name, module in model.named_modules():
        if hasattr(module, 'alpha'):
            # lets get layer num from the name
            layer_num = int(name.split('.')[1]) if 'encoder' in  name else int(name.split('.')[1]) + model.encoder_layers
            alpha_values = module.alpha.detach().cpu().numpy()

            for head_idx, alpha in enumerate(alpha_values):
                alphas.append(alpha)
                head_positions.append(head_idx)
                layer_number.append(layer_num)

    # creating the heatmap
    alpha_matrix = np.zeros((max(layer_number) + 1, max(head_positions) + 1))
    for l, h, a in zip(layer_number, head_positions, alphas):
        alpha_matrix[l, h] = a
    
    plt.figure(figsize=(10,8))
    sns.heatmap(alpha_matrix, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Learned alpha values across layers and heads')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.savefig(f"{save_dir}/alpha_values.png")
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