import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from train import get_model, get_ds, train_model
from config import get_config

def setup(rank, world_size, dist_backend, dist_url):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_distributed(rank, world_size, config):
    # Setup distributed training
    setup(rank, world_size, config['dist_backend'], config['dist_url'])
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Get dataloaders with DistributedSampler
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    train_sampler = DistributedSampler(train_dataloader.dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataloader.dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler
    )
    
    # Create model and move it to GPU with DDP
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    # Update config with current GPU
    config['gpu'] = rank
    
    # Train the model
    train_model(config, model, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt)
    
    cleanup()

if __name__ == "__main__":
    # Get config
    config = get_config()
    
    # Determine number of GPUs
    if config['world_size'] == -1:
        config['world_size'] = torch.cuda.device_count()
    
    print(f"Using {config['world_size']} GPUs for training")
    
    # Launch processes
    mp.spawn(
        train_distributed,
        args=(config['world_size'], config),
        nprocs=config['world_size'],
        join=True
    )