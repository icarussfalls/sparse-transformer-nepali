import os
import sys
import contextlib

# Redirect stderr to suppress CUDA warnings
@contextlib.contextmanager
def suppress_cuda_warnings():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Set environment variables before other imports
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NCCL_DEBUG"] = "NONE"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

# Import torch with suppressed warnings
with suppress_cuda_warnings():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader

from train import get_model, get_ds, train_model
from config import get_config
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
for logger_name in ['torch', 'torch.distributed', 'torch.nn.parallel', 'torch.cuda', 'torch.utils.data']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def collate_fn(batch):
    """Convert list of samples to dictionary batch"""
    return {
        'encoder_input': torch.stack([item['encoder_input'] for item in batch]),
        'decoder_input': torch.stack([item['decoder_input'] for item in batch]),
        'encoder_mask': torch.stack([item['encoder_mask'] for item in batch]),
        'decoder_mask': torch.stack([item['decoder_mask'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    with suppress_cuda_warnings():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training environment"""
    dist.destroy_process_group()

def train_ddp(rank, world_size, config):
    try:
        # Initialize distributed training
        setup(rank, world_size)
        
        # Update config for this GPU
        config['gpu'] = rank
        config['rank'] = rank
        config['world_size'] = world_size
        
        # Add DDP suffix to model folder to avoid conflicts
        config['model_folder'] = f"{config['model_folder']}_ddp"
        
        # Just call train_model with config - it handles everything internally
        from train import train_model
        train_model(config)
    
    finally:
        cleanup()

def main():
    try:
        # Disable CUDA warnings
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Get config
        config = get_config()
        
        # Get world size (number of GPUs)
        world_size = torch.cuda.device_count()
        print(f"Using {world_size} GPUs")
        
        # Launch processes
        mp.spawn(
            train_ddp,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    main()