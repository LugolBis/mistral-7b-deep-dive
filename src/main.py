import torch

from model import RMSNorm, RotaryPositionalEmbedding, TransformerBlock


if __name__ == "__main__":
    norm = RMSNorm(dim=4)
    sample = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(norm(sample))

    rope = RotaryPositionalEmbedding(head_dim=4)
    positional_sample = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    print(f"RoPE output: {rope(positional_sample, torch.tensor([1]))}")

    block = TransformerBlock(
        dim=128,
        n_heads=4,
        n_kv_heads=2,
        window_size=4,
        hidden_dim=256,
    )
    tokens = torch.randn(1, 6, 128)
    print(f"Transformer block output shape: {block(tokens).shape}")
