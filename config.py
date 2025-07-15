from pathlib import Path

def get_config():
    return {
        'batch_size': 64,
        'val_batch_size': 64,
        'gradient_accumulation_steps': 1,  # update weights every step
        'gradient_checkpointing': False,  
        'num_epochs' : 20,
        'lr': 10**-4,
        'seq_len': 200,
        'd_model' : 512,
        'd_ff' : 1024,
        'N': 4, 
        'h': 4,
        'dropout': 0.1,
        'use_sparse': False,
        'sparse_block_size': 64,
        'sparse_stride': 64,
        'use_adaptive_sparse': True, 
        'attn_type': "sparsemax", # or "sparsemax"
        'visualize': True,  # Set to False for normal training
        'visualize_frequency': 10,  # Only visualize every 10 epochs
        'data_source': 'sharad461/ne-en-parallel-208k',
        'lang_src': 'en',
        'lang_tgt': 'ne',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': 'latest',
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel',
        'distributed': True,
        'world_size': -1,
        'dist_backend': 'nccl',
        'dist_url': 'tcp://localhost:23456',
        'gpu': None,
    }


def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    return f"{model_folder}/{model_basename}_{epoch}.pt"

# find the latest weights file in the weights folder
def latest_weight_file_path(config):
    model_folder = config['model_folder'] 
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def auto_configure_paths(config):
    """
    Automatically sets model_folder and experiment_name in config 
    based on architecture flags that are already set.
    
    Call this right after creating a config but before using it.
    """
    # Determine architecture based on flags
    if config.get('use_sparse', False):
        architecture = 'sparse'
        description = 'Fixed Sparse Attention'
        model_folder = 'weights_sparse'
        experiment_name = 'runs/sparse'
    elif config.get('use_adaptive_sparse', False):
        architecture = 'adaptive_sparse'
        description = 'Adaptive Sparse Attention'
        model_folder = 'weights_adaptive_sparse'
        experiment_name = 'runs/adaptive_sparse'
    else:
        architecture = 'vanilla'
        description = 'Standard Transformer'
        model_folder = 'weights_vanilla'
        experiment_name = 'runs/vanilla'
    
    # Update config with architecture-specific paths
    config['model_folder'] = model_folder
    config['experiment_name'] = experiment_name
    config['architecture'] = architecture
    config['description'] = description
    
    print(f"Configured for {architecture.upper()}: {description}")
    print(f"Model folder: {model_folder}")
    print(f"Experiment name: {experiment_name}")
    
    return config