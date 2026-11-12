# Tensor & Computation Explorer

A standalone, dependency-free browser prototype for inspecting LLM tensor computation. The first vertical slice models Multi-head Latent Attention (MLA).

## Run

From this directory:

```bash
python -m http.server 8011
```

Open http://127.0.0.1:8011/

No bundler, package installation, backend, or GPU is required.

## Tests

Requires Node.js 18+:

```bash
npm test
```

## Scope

- Model Overview is intentionally deferred; this MVP focuses on the local computation-flow view.
- Tensor and Operation nodes are generated from the same computation model.
- Shape expressions are resolved from editable model configuration.
- FLOPs use 2 FLOPs per multiply-accumulate.
- Weight and logical tensor memory are computed from declared shapes and dtype.
- KV cache estimate assumes per layer storage of compressed KV latent plus shared RoPE key.
- Prefill uses Q length = KV length = sequence length.
- Decode uses Q length = 1 and KV length = configured sequence length.
- Softmax, RoPE, normalization and other elementwise costs are excluded from headline FLOPs.
- The graph shows the explicit K/V materialization path. Absorbed MLA decode variants are noted but are not yet represented as a separate graph.

## Important limitations

This is an analytical architecture model, not a profiler. It does not estimate kernel workspace, allocator fragmentation, activation liveness peak, backward-pass memory, communication, or runtime latency. Weight shapes are shown in mathematical matrix orientation; a framework may store transposed weight tensors.
