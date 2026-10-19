# mistral-7b-deep-dive

This project is a deep dive into the research paper *Mistral 7B* (2023). It aims to explain, in accessible terms, why a 7-billion-parameter model can compete with much larger ones, by breaking down the architectural choices and optimizations described in the paper : Grouped-Query Attention, Sliding Window Attention, the rolling buffer cache, and pre-fill with chunking. The repository contains the slides of an oral presentation that walks through these ideas, with diagrams and worked examples.

The project is also building an educational, from-scratch Python reimplementation
of the Mistral 7B decoder. It is currently in progress: token embeddings,
RMSNorm, RoPE, causal Grouped-Query Sliding-Window Attention, SwiGLU, and a
pre-normalized Transformer block with residual connections are implemented,
along with an educational decoder that produces next-token logits and performs
greedy generation with a per-layer rolling KV cache. The cache keeps at most one
sliding window of keys and values, so generation only processes each new token
after the initial prompt. Pass `use_cache=False` to `generate` to recompute the
full context for comparison. The prompt can also be prefilled in smaller
chunks with `prefill_chunk_size`, while preserving its absolute token positions
and reusing the rolling cache between chunks. The project does not yet include
a tokenizer, and its weights are randomly initialized rather than pretrained.

## Repository structure

```text
.
├── doc/             # Presentation slides
├── src/             # Decoder reimplementation
└── README.md
```

## Installation

```bash
git clone https://github.com/LugolBis/mistral-7b-deep-dive.git
cd mistral-7b-deep-dive
uv venv && uv sync
```

## Usage

The presentation could be compiled as *PDF* using :
```bash
typst compile doc/presentation.typ
```

To run the miniature decoder example:
```bash
uv run python src/main.py
```

To run its tests:
```bash
uv run python -m unittest discover -s tests
```