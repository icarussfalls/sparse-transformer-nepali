import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from train import get_model, get_ds, train_model
from config import get_config

# Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training environment"""
    dist.destroy_process_group()

def train_ddp(rank, world_size, config):
    try:
        # Initialize distributed training
        setup(rank, world_size)
        
        # Get data and model
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
        
        # Create DDP-compatible dataloaders
        train_sampler = DistributedSampler(
            train_dataloader.dataset,
            num_replicas=world_size,
            rank=rank
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataloader.dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model and move to GPU
        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
        if config['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        model = model.to(rank)
        
        # Wrap model in DDP with memory-efficient settings
        ddp_model = DDP(
            model, 
            device_ids=[rank],
            find_unused_parameters=False,  # Disable if not needed
            static_graph=True  # Enable for better memory efficiency
        )
        
        # Update config with current GPU
        config['gpu'] = rank
        
        # Train the model
        train_model(
            config, 
            model=ddp_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt
        )
    
    finally:
        cleanup()  # Ensure cleanup happens even if training fails

def main():
    try:
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