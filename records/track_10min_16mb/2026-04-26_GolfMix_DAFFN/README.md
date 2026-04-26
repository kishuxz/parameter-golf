# GolfMix_DAFFN: Leaderboard-focused Parameter Golf stack with optional Depth-Adaptive FFN allocation

## 1. Summary

GolfMix_DAFFN is a leaderboard-focused 10-minute / 16MB Parameter Golf attempt. By default it preserves the strongest reproducible public record found locally, then exposes controlled ablations for Depth-Adaptive FFN allocation and scalar StarReLU.

Default settings are base-preserving:

- `USE_DAFFN=0`
- `MLP_SCHEDULE=uniform`
- `USE_STARRELU=0`
- `USE_NGRAM_MIX=0`

Final settings should be chosen empirically from validation `val_bpb`; placeholders below use `X.XXXX` until RunPod experiments are complete.

## 2. Base implementation selected

Base script:

`records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`

Selection reason:

- Best local public 10-minute record inspected: mean `val_bpb=1.14276` over 3 seeds.
- Reproducible record folder with README, submission metadata, and seed logs.
- Already fits close to the 16MB limit using mixed Int5/Int6 compression.
- GPT-based and lower risk than the experimental SSM path.
- Compatible with a controlled MLP allocation ablation.

## 3. Techniques reused from public leaderboard approaches

- SentencePiece SP1024 tokenizer and tokenizer-agnostic BPB calculation.
- 10-layer GPT, 512 model dimension, 8 query heads, 4 KV heads.
- Grouped-query causal self-attention with RoPE and QK gain.
- U-Net style skip connections.
- SmearGate embedding mixing.
- BigramHash embedding with 10240 buckets.
- 3x ReLU-squared MLP expansion.
- Orthogonal initialization with scaled output projections.
- Muon optimizer with tuned weight decay and momentum warmup.
- AdamW for token/scalar parameters.
- Stochastic Weight Averaging.
- 3% magnitude pruning before export.
- Mixed compression: Int5 MLP, Int6 attention/bigram, FP16 for sensitive tensors, zstd-22 when available.
- Sliding-window validation evaluation with stride 64.

## 4. Unique contribution: DAFFN-Star

DAFFN-Star is optional and off by default.

`USE_DAFFN=1` enables depth-adaptive MLP hidden widths while keeping average MLP multiplier close to the base target. Supported schedules:

- `uniform`
- `linear_increasing`
- `linear_decreasing`
- `middle_heavy`
- `alternating`

Relevant env vars:

- `MLP_MULT_MIN=2.0`
- `MLP_MULT_MAX=4.0`
- `MLP_MULT_AVG=3.0`

`USE_STARRELU=1` replaces compatible `relu(x)^2` activations with scalar StarReLU:

```python
scale * relu(x)^2 + bias
```

The scale and bias are scalar learned parameters per MLP block, not large vectors.

## 5. Optional n-gram mixture

`USE_NGRAM_MIX=0` by default.

This record includes a scaffold and explicit legality logging only. It does not alter BPB scoring, because a safe implementation must build tables only from allowed training tokens and account for artifact size. Validation tokens must never be used to build or update n-gram tables before scoring.

## 6. Legality and no validation leakage statement

The default implementation trains only from configured training shards and evaluates on validation shards without training on validation tokens. Sliding-window evaluation uses only past context within each validation window and does not expose future target tokens. The optional n-gram path is scaffold-only and leaves neural-only scoring unchanged.

No network calls are used during training or evaluation.

## 7. Architecture

- Tokenizer: SentencePiece SP1024 by default.
- Layers: 10.
- Dimension: 512.
- Heads: 8 query heads, 4 KV heads.
- Attention: causal GQA with RoPE and QK gain.
- MLP: base 3x hidden width, optional DAFFN schedule.
- Embeddings: tied token embeddings plus BigramHash and SmearGate.
- Recurrence: none.
- XSA: none.
- Skip structure: encoder/decoder U-Net style skip connections.

The script logs exact per-layer MLP hidden sizes and multipliers at startup.

## 8. Compression/artifact size

The base uses mixed quantization:

- MLP matrices: Int5.
- Attention and BigramHash matrices: Int6.
- Sensitive tensors such as token embeddings and selected late attention weights: FP16.
- Small control/scalar tensors: passthrough.
- Compression: zstd-22 if installed, otherwise zlib-9.

The script logs compressed model size, estimated code+model size, and PASS/FAIL against 16,000,000 bytes.

## 9. How to run

From this folder:

```bash
mkdir -p logs
python3 -m py_compile train_gpt.py
MODEL_INIT_CHECK=1 python3 train_gpt.py
```

Then run the smoke or full commands in `run_commands.md`.

## 10. Ablation table template

| Run | USE_DAFFN | MLP_SCHEDULE | USE_STARRELU | USE_NGRAM_MIX | Seed | val_bpb | artifact bytes | Notes |
|-----|-----------|--------------|--------------|----------------|------|---------|----------------|-------|
| Base smoke | 0 | uniform | 0 | 0 | 42 | X.XXXX | X | 120s smoke |
| Base full | 0 | uniform | 0 | 0 | 42 | X.XXXX | X | Base reproduction |
| DAFFN middle | 1 | middle_heavy | 0 | 0 | 42 | X.XXXX | X | DAFFN only |
| StarReLU only | 0 | uniform | 1 | 0 | 42 | X.XXXX | X | StarReLU only |
| DAFFN-Star | 1 | middle_heavy | 1 | 0 | 42 | X.XXXX | X | Combined |

## 11. Final results table template

| Seed | Setting | val_loss | val_bpb | compressed bytes | total estimate | PASS/FAIL |
|------|---------|----------|---------|------------------|----------------|-----------|
| 42 | BEST_VALUE_HERE | X.XXXX | X.XXXX | X | X | X |
| 1337 | BEST_VALUE_HERE | X.XXXX | X.XXXX | X | X | X |
| 2024 | BEST_VALUE_HERE | X.XXXX | X.XXXX | X | X | X |
| Mean | BEST_VALUE_HERE | X.XXXX | X.XXXX | X | X | X |

## 12. Known limitations

- Default tokenizer remains SP1024; larger SP4096/SP8192 tokenizer work is not included here.
- XSA and depth recurrence are not implemented in this record.
- DAFFN may change compression ratio even if parameter count is near the base average, so artifact size must be checked for every ablation.
- N-gram mixture is scaffold-only until training-only table construction, smoothing, blending, and artifact accounting are fully verified.
