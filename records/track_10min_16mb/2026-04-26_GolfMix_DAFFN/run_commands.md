# GolfMix_DAFFN Run Commands

Run from:

```bash
cd records/track_10min_16mb/2026-04-26_GolfMix_DAFFN
mkdir -p logs
```

## A. Compile check

```bash
python3 -m py_compile train_gpt.py
```

## B. 1xH100 smoke test, base only

```bash
RUN_ID=golfmix_base_smoke \
USE_DAFFN=0 \
USE_STARRELU=0 \
USE_NGRAM_MIX=0 \
MAX_WALLCLOCK_SECONDS=120 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/golfmix_base_smoke.log
```

## C. 1xH100 full base reproduction

```bash
RUN_ID=golfmix_base_1xh100 \
USE_DAFFN=0 \
USE_STARRELU=0 \
USE_NGRAM_MIX=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/golfmix_base_1xh100.log
```

## D. DAFFN middle-heavy ablation

```bash
RUN_ID=golfmix_daffn_middle \
USE_DAFFN=1 \
MLP_SCHEDULE=middle_heavy \
USE_STARRELU=0 \
USE_NGRAM_MIX=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/golfmix_daffn_middle.log
```

## E. StarReLU only

```bash
RUN_ID=golfmix_starrelu_only \
USE_DAFFN=0 \
USE_STARRELU=1 \
USE_NGRAM_MIX=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/golfmix_starrelu_only.log
```

## F. DAFFN + StarReLU

```bash
RUN_ID=golfmix_daffn_starrelu \
USE_DAFFN=1 \
MLP_SCHEDULE=middle_heavy \
USE_STARRELU=1 \
USE_NGRAM_MIX=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/golfmix_daffn_starrelu.log
```

## G. Optional n-gram mix scaffold

This command currently logs the n-gram configuration and keeps neural-only scoring. Do not use it as a final mixed scorer until training-only table construction and artifact accounting are implemented and verified.

```bash
RUN_ID=golfmix_ngram_mix \
USE_DAFFN=0 \
USE_STARRELU=0 \
USE_NGRAM_MIX=1 \
NGRAM_MAX_ORDER=7 \
NGRAM_ALPHA_BASE=0.10 \
NGRAM_ALPHA_MAX=0.50 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/golfmix_ngram_mix.log
```

## H. 8xH100 final seed commands

```bash
RUN_ID=golfmix_final_seed42 \
SEED=42 \
USE_DAFFN=BEST_VALUE_HERE \
MLP_SCHEDULE=BEST_VALUE_HERE \
USE_STARRELU=BEST_VALUE_HERE \
USE_NGRAM_MIX=BEST_VALUE_HERE \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/golfmix_final_seed42.log
```

```bash
RUN_ID=golfmix_final_seed1337 \
SEED=1337 \
USE_DAFFN=BEST_VALUE_HERE \
MLP_SCHEDULE=BEST_VALUE_HERE \
USE_STARRELU=BEST_VALUE_HERE \
USE_NGRAM_MIX=BEST_VALUE_HERE \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/golfmix_final_seed1337.log
```

```bash
RUN_ID=golfmix_final_seed2024 \
SEED=2024 \
USE_DAFFN=BEST_VALUE_HERE \
MLP_SCHEDULE=BEST_VALUE_HERE \
USE_STARRELU=BEST_VALUE_HERE \
USE_NGRAM_MIX=BEST_VALUE_HERE \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/golfmix_final_seed2024.log
```
