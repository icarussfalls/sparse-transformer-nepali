import torch
from adaptive_sparse_model import build_adaptive_sparse_transformer
from visualization import visualize_alpha_values

def test_alpha_visualization():
    """Test alpha visualization with a fresh model"""
    # Create a small model for testing
    model = build_adaptive_sparse_transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        src_seq_len=128,
        tgt_seq_len=128,
        d_model=256,
        N=2,  # 2 layers
        h=4,  # 4 heads
        dropout=0.1,
        d_ff=512,
        attn_type="entmax_alpha"
    )
    
    # Initialize some dummy alpha values
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and module.alpha is not None:
            # Initialize with some variation
            module.alpha.data.uniform_(1.0, 2.0)
    
    # Generate visualization
    visualize_alpha_values(model, 'test_alpha_viz')
    print("Test alpha visualization saved to test_alpha_viz/alpha_distributions.png")

if __name__ == "__main__":
    test_alpha_visualization()