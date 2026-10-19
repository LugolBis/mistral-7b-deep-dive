import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root mean square normalization over the final tensor dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("RMSNorm input must have a floating-point dtype")
        if x.ndim == 0:
            raise ValueError("RMSNorm input must have at least one dimension")
        if x.shape[-1] != self.weight.numel():
            raise ValueError(
                f"Expected the last dimension to be {self.weight.numel()}, "
                f"got {x.shape[-1]}"
            )

        input_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        x_for_norm = x.to(dtype=compute_dtype)
        mean_square = x_for_norm.square().mean(dim=-1, keepdim=True)
        normalized = x_for_norm * torch.rsqrt(mean_square + self.eps)

        return normalized.to(dtype=input_dtype) * self.weight.to(dtype=input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary position embeddings to attention heads."""

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even number")
        if theta <= 0:
            raise ValueError("theta must be positive")

        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("RoPE input must have a floating-point dtype")
        if x.ndim != 4:
            raise ValueError("RoPE input must have shape (batch, heads, seq, head_dim)")
        batch_size, _, seq_len, head_dim = x.shape
        if head_dim != self.head_dim:
            raise ValueError(
                f"Expected the last dimension to be {self.head_dim}, got {head_dim}"
            )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
        else:
            if position_ids.ndim != 1 or position_ids.numel() != seq_len:
                raise ValueError("position_ids must have shape (seq_len,)")
            if position_ids.is_floating_point() or position_ids.is_complex():
                raise TypeError("position_ids must have an integer dtype")
            if position_ids.device != x.device:
                raise ValueError("position_ids and input must be on the same device")
            if torch.any(position_ids < 0):
                raise ValueError("position_ids must be non-negative")

        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        pair_indices = torch.arange(
            0, self.head_dim, 2, device=x.device, dtype=compute_dtype
        )
        inverse_frequencies = self.theta ** (-pair_indices / self.head_dim)
        angles = position_ids.to(dtype=compute_dtype)[:, None] * inverse_frequencies
        cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]

        x_pairs = x.reshape(batch_size, x.shape[1], seq_len, self.head_dim // 2, 2)
        even = x_pairs[..., 0]
        odd = x_pairs[..., 1]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        )
        return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    """Causal sliding-window attention with grouped key/value heads."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        window_size: int,
    ) -> None:
        super().__init__()
        if dim <= 0 or n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("dim and head counts must be positive")
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        head_dim = dim // n_heads
        if head_dim % 2 != 0:
            raise ValueError("head dimension must be even for rotary embeddings")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.rope = RotaryPositionalEmbedding(self.head_dim)

        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("attention input must have a floating-point dtype")
        if x.ndim != 3:
            raise ValueError("attention input must have shape (batch, seq, dim)")
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected input dimension {self.dim}, got {dim}")
        if seq_len == 0:
            raise ValueError("attention input must contain at least one token")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
        elif position_ids.ndim != 1 or position_ids.numel() != seq_len:
            raise ValueError("position_ids must have shape (seq_len,)")
        elif position_ids.is_floating_point() or position_ids.is_complex():
            raise TypeError("position_ids must have an integer dtype")
        elif position_ids.device != x.device:
            raise ValueError("position_ids and input must be on the same device")
        elif torch.any(position_ids < 0):
            raise ValueError("position_ids must be non-negative")

        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        groups_per_kv_head = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(groups_per_kv_head, dim=1)
        v = v.repeat_interleave(groups_per_kv_head, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        query_positions = position_ids[:, None]
        key_positions = position_ids[None, :]
        allowed = (key_positions <= query_positions) & (
            key_positions > query_positions - self.window_size
        )
        scores = scores.masked_fill(
            ~allowed[None, None, :, :], torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        return self.out_proj(attended)


class SwiGLU(nn.Module):
    """Gated feed-forward network used in the Transformer block."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("SwiGLU input must have a floating-point dtype")
        if x.ndim == 0 or x.shape[-1] != self.gate_proj.in_features:
            raise ValueError(
                f"Expected input's last dimension to be {self.gate_proj.in_features}"
            )

        gate = torch.nn.functional.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)


class TransformerBlock(nn.Module):
    """Pre-normalized Mistral-style attention and feed-forward block."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        window_size: int,
        hidden_dim: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")

        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = GroupedQueryAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            window_size=window_size,
        )
        self.feed_forward_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = SwiGLU(dim=dim, hidden_dim=hidden_dim)

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("TransformerBlock input must have a floating-point dtype")
        if x.ndim != 3:
            raise ValueError("TransformerBlock input must have shape (batch, seq, dim)")

        x = x + self.attention(self.attention_norm(x), position_ids)
        return x + self.feed_forward(self.feed_forward_norm(x))


class TokenEmbedding(nn.Module):
    """Map token IDs to learned vectors."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        if vocab_size <= 0 or dim <= 0:
            raise ValueError("vocab_size and dim must be positive")

        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, seq)")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must have dtype torch.int32 or torch.int64")

        return self.embedding(token_ids)


class MiniMistralDecoder(nn.Module):
    """Educational decoder that maps token IDs to next-token logits."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        window_size: int,
        hidden_dim: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")

        self.vocab_size = vocab_size
        self.dim = dim
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, dim=dim)
        self.layers = nn.ModuleList(
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                window_size=window_size,
                hidden_dim=hidden_dim,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        )
        self.final_norm = RMSNorm(dim, eps=norm_eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self, token_ids: Tensor, position_ids: Tensor | None = None
    ) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, seq)")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must have dtype torch.int32 or torch.int64")

        hidden_states = self.token_embedding(token_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)

    @torch.inference_mode()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Greedily append tokens, recomputing the full context at each step."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, seq)")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must have dtype torch.int32 or torch.int64")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if eos_token_id is not None and not 0 <= eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id must be within the vocabulary")

        generated = token_ids.clone()
        finished = torch.zeros(
            token_ids.shape[0], dtype=torch.bool, device=token_ids.device
        )

        for _ in range(max_new_tokens):
            positions = torch.arange(generated.shape[1], device=generated.device)
            logits = self(generated, position_ids=positions)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished |= next_token.squeeze(-1).eq(eos_token_id)

            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and finished.all():
                break

        return generated
