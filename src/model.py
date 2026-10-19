import math

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


class KVCache:
    """Per-layer rolling key/value cache with a fixed maximum length."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self.max_size = max_size
        self._layers: dict[int, tuple[Tensor, Tensor, Tensor]] = {}

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor, Tensor] | None:
        """Return cached keys, values, and their absolute positions."""
        return self._layers.get(layer_idx)

    def update(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
        position_ids: Tensor,
    ) -> None:
        if layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("keys and values must have matching (batch, heads, seq, dim) shapes")
        if position_ids.ndim != 1 or position_ids.numel() != keys.shape[2]:
            raise ValueError("position_ids must have shape (seq_len,)")
        if position_ids.is_floating_point() or position_ids.is_complex():
            raise TypeError("position_ids must have an integer dtype")
        if position_ids.device != keys.device or values.device != keys.device:
            raise ValueError("keys, values, and positions must use the same device")
        if keys.dtype != values.dtype:
            raise ValueError("keys and values must have the same dtype")
        if position_ids.numel() == 0:
            raise ValueError("cache updates must contain at least one position")
        if torch.any(position_ids < 0):
            raise ValueError("position_ids must be non-negative")
        if position_ids.numel() > 1 and torch.any(position_ids[1:] <= position_ids[:-1]):
            raise ValueError("position_ids in a cache update must be strictly increasing")

        previous = self._layers.get(layer_idx)
        if previous is not None:
            previous_keys, previous_values, previous_positions = previous
            if keys.shape[:2] != previous_keys.shape[:2] or keys.shape[3] != previous_keys.shape[3]:
                raise ValueError("cache update has incompatible key/value dimensions")
            if keys.dtype != previous_keys.dtype or keys.device != previous_keys.device:
                raise ValueError("cache update must match the cached dtype and device")
            if position_ids[0] <= previous_positions[-1]:
                raise ValueError("cache positions must advance beyond the previous update")
            keys = torch.cat((previous_keys, keys), dim=2)
            values = torch.cat((previous_values, values), dim=2)
            position_ids = torch.cat((previous_positions, position_ids))

        self._layers[layer_idx] = (
            keys[:, :, -self.max_size :],
            values[:, :, -self.max_size :],
            position_ids[-self.max_size :],
        )

    def clear(self) -> None:
        """Remove cached state for all layers."""
        self._layers.clear()


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

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("attention input must have a floating-point dtype")
        if x.ndim != 3:
            raise ValueError("attention input must have shape (batch, seq, dim)")
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected input dimension {self.dim}, got {dim}")
        if seq_len == 0:
            raise ValueError("attention input must contain at least one token")
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when using a KV cache")
            if kv_cache.max_size != self.window_size:
                raise ValueError("KV cache size must match the attention window")
        if position_ids is None:
            start_position = 0
            if kv_cache is not None:
                cached = kv_cache.get(layer_idx)
                if cached is not None:
                    start_position = int(cached[2][-1].item()) + 1
            position_ids = torch.arange(
                start_position, start_position + seq_len, device=x.device
            )
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
        new_k, new_v = k, v

        key_positions = position_ids
        if kv_cache is not None:
            cached = kv_cache.get(layer_idx)
            if cached is not None:
                cached_k, cached_v, cached_positions = cached
                if cached_k.shape[:2] != k.shape[:2] or cached_k.shape[3] != self.head_dim:
                    raise ValueError("KV cache dimensions do not match this attention layer")
                if cached_k.dtype != k.dtype or cached_k.device != k.device:
                    raise ValueError("KV cache dtype and device must match the attention input")
                k = torch.cat((cached_k, k), dim=2)
                v = torch.cat((cached_v, v), dim=2)
                key_positions = torch.cat((cached_positions, position_ids))

        groups_per_kv_head = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(groups_per_kv_head, dim=1)
        v = v.repeat_interleave(groups_per_kv_head, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        query_positions = position_ids[:, None]
        key_positions_for_mask = key_positions[None, :]
        allowed = (key_positions_for_mask <= query_positions) & (
            key_positions_for_mask > query_positions - self.window_size
        )
        scores = scores.masked_fill(
            ~allowed[None, None, :, :], torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        output = self.out_proj(attended)
        if kv_cache is not None:
            kv_cache.update(layer_idx, new_k, new_v, position_ids)
        return output


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

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("TransformerBlock input must have a floating-point dtype")
        if x.ndim != 3:
            raise ValueError("TransformerBlock input must have shape (batch, seq, dim)")

        x = x + self.attention(
            self.attention_norm(x), position_ids, kv_cache, layer_idx
        )
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
        self,
        token_ids: Tensor,
        position_ids: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, seq)")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must have dtype torch.int32 or torch.int64")

        hidden_states = self.token_embedding(token_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_ids,
                kv_cache,
                layer_idx if kv_cache is not None else None,
            )
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)

    @torch.inference_mode()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        prefill_chunk_size: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate greedily or sample using temperature, top-k, and top-p."""
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
        if prefill_chunk_size is not None and prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if prefill_chunk_size is not None and not use_cache:
            raise ValueError("chunked prefill requires use_cache=True")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite, non-negative number")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be a finite number in (0, 1]")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")

        generated = token_ids.clone()
        if max_new_tokens == 0:
            return generated

        kv_cache = (
            KVCache(self.layers[0].attention.window_size) if use_cache else None
        )
        chunk_size = prefill_chunk_size or generated.shape[1]
        for start in range(0, generated.shape[1], chunk_size):
            end = min(start + chunk_size, generated.shape[1])
            positions = torch.arange(start, end, device=generated.device)
            logits = self(
                generated[:, start:end],
                position_ids=positions,
                kv_cache=kv_cache,
            )
        finished = torch.zeros(
            token_ids.shape[0], dtype=torch.bool, device=token_ids.device
        )

        for step in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                if generator is not None and generator.device != next_token_logits.device:
                    raise ValueError(
                        "sampling generator and logits must use the same device"
                    )
                if top_k is not None:
                    top_indices = next_token_logits.topk(
                        min(top_k, next_token_logits.shape[-1]), dim=-1
                    ).indices
                    filtered_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )
                    next_token_logits = filtered_logits.scatter(
                        -1, top_indices, next_token_logits.gather(-1, top_indices)
                    )
                probabilities = torch.softmax(
                    next_token_logits / temperature, dim=-1
                )
                if top_p < 1:
                    sorted_probabilities, sorted_indices = probabilities.sort(
                        dim=-1, descending=True
                    )
                    cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
                    remove_sorted = (
                        cumulative_probabilities - sorted_probabilities >= top_p
                    )
                    filtered_sorted_probabilities = sorted_probabilities.masked_fill(
                        remove_sorted, 0.0
                    )
                    probabilities = torch.zeros_like(probabilities).scatter(
                        -1, sorted_indices, filtered_sorted_probabilities
                    )
                    probabilities = probabilities / probabilities.sum(
                        dim=-1, keepdim=True
                    )
                next_token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )

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
            if step + 1 < max_new_tokens:
                if kv_cache is None:
                    positions = torch.arange(
                        generated.shape[1], device=generated.device
                    )
                    logits = self(generated, position_ids=positions)
                else:
                    position = torch.tensor(
                        [generated.shape[1] - 1], device=generated.device
                    )
                    logits = self(
                        next_token, position_ids=position, kv_cache=kv_cache
                    )

        return generated
