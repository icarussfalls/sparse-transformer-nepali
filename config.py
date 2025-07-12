from pathlib import Path

def get_config():
    return {
        'batch_size': 64,
        'val_batch_size': 64,
        'gradient_accumulation_steps': 1,  # update weights every step
        'gradient_checkpointing': False,  # Keep disabled for speed
        'num_epochs' : 20,
        'lr': 10**-4,
        'seq_len': 200,
        'd_model' : 512,
        'd_ff' : 2048,
        'N': 4, 
        'h': 4,
        'dropout': 0.1,
        'use_sparse': False,
        'use_adaptive_sparse': True, 
        'attn_type': "entmax_alpha",
        'visualize': True, 
        'sparse_block_size': 64,
        'sparse_stride': 64,
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


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['data_source']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# find the latest weights file in the weights folder
def latest_weight_file_path(config):
    model_folder = f"{config['data_source']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    # this searches all the weight files and return all with name
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
