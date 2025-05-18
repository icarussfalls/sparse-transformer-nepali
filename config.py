from pathlib import Path

def get_config():
    return {
        'batch_size': 8,
        'num_epochs' : 20,
        'lr': 10**-4,
        'seq_len': 600,
        'd_model' : 512,
        'data_source': 'sharad461/ne-en-parallel-208k',
        'lang_src': 'en',
        'lang_tgt': 'ne',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': 'latest',
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
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
