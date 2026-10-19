import torch

from model import MiniMistralDecoder, RMSNorm, RotaryPositionalEmbedding


if __name__ == "__main__":
    norm = RMSNorm(dim=4)
    sample = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(norm(sample))

    rope = RotaryPositionalEmbedding(head_dim=4)
    positional_sample = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    print(f"RoPE output: {rope(positional_sample, torch.tensor([1]))}")

    model = MiniMistralDecoder(
        vocab_size=32_000,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        window_size=4,
        hidden_dim=256,
    )
    token_ids = torch.tensor([[12, 95, 7, 33, 90, 11]])
    logits = model(token_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Highest-scoring next token ID: {logits[0, -1].argmax().item()}")
    generated = model.generate(token_ids, max_new_tokens=5)
    print(f"Generated token IDs: {generated[0].tolist()}")
