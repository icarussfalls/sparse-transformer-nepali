from pathlib import Path

def get_config():
    return {
        'batch_size': 8,
        'num_epochs': 10,
        'lr': 10**-4,
        'seq_len': 350,
        'd_model': 512,
        'lang_src': "en",
        'lang_tgt': "ne",
        'data_source': 'sharad461/ne-en-parallel-208k',
        'model_folder': "weights",  # Base folder, will be modified per architecture
        'model_basename': "tmodel",
        'preload': False,
        'tokenizer_file': "tokenizer_{0}.json",
        'experiment_name': "runs/tmodel",
        'N': 6,
        'h': 8,
        'dropout': 0.1,
        'd_ff': 2048,
        'use_sparse': False,
        'use_adaptive_sparse': False,
        'sparse_block_size': 64,
        'sparse_stride': 32,
        'attn_type': 'adaptive',
        'val_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'warmup_steps': 1000
    }


def get_weights_file_path(config, epoch):
    """Get weights file path for specific architecture"""
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    return f"{model_folder}/{model_basename}_{epoch}.pt"

# find the latest weights file in the weights folder
def latest_weight_file_path(config):
    """Get latest weights file for specific architecture"""
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    
    # Look for files in architecture-specific folder
    model_files = list(Path(model_folder).glob(f"{model_basename}_*.pt"))
    if not model_files:
        return None
    
    # Get the latest file
    latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
    return str(latest_file)

def get_architecture_config(arch_type):
    """Get architecture-specific config"""
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
