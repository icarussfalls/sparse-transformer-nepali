import os
# Set environment variables before other imports
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
# Add these new environment variables
os.environ["NCCL_DEBUG"] = "NONE"  # Disable NCCL debugging
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"  # Changed from "NONE" to "OFF"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

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

# Configure logging more aggressively
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.nn.parallel").setLevel(logging.ERROR)
logging.getLogger("torch.cuda").setLevel(logging.ERROR)
logging.getLogger("torch.utils.data").setLevel(logging.ERROR)

# Silence warnings
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

# Set CUDA environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TensorFlow CUDA warnings

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
        
        # Create model and move to GPU
        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
        
        # Enable gradient checkpointing if configured
        if config['gradient_checkpointing']:
            print(f"Rank {rank}: Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()
            
        model = model.to(rank)
        
        # Wrap model in DDP
        ddp_model = DDP(
            model, 
            device_ids=[rank],
            find_unused_parameters=False,
            static_graph=True
        )
        
        # Update config with current GPU
        config['gpu'] = rank
        
        train_sampler = DistributedSampler(train_dataloader.dataset, num_replicas=world_size, rank=rank)
        
        train_dataloader = DataLoader(
            train_dataloader.dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=True,
            drop_last=True,
            collate_fn=collate_fn  # Use the moved collate function
        )
        
        # Create validation dataloader with DDP
        val_sampler = DistributedSampler(
            val_dataloader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False  # No need to shuffle validation
        )
        
        val_dataloader = DataLoader(
            val_dataloader.dataset,
            batch_size=config['val_batch_size'],
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=True,
            drop_last=True,
            collate_fn=collate_fn  # Add custom collate function
        )
        
        # Enable cudNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
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