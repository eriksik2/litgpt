# Learned Self-State Vector

This fork adds an explicit **self-state** stream to LitGPT so the model can keep a compact summary of its own intermediate computation and use it for monitoring.

## Idea

```
tokens
  ↓
Transformer block
  ↓
activations
  ↓
self-state ──────┐
  ↓              │
next block ←─────┘
  ↓
...
  ↓
prediction
```

After each Transformer block:

1. Encode activations into a candidate state: `new_state = encoder(activations)`
2. Gated update: `self_state = gate ⊙ old_state + (1 - gate) ⊙ new_state`
3. Inject into the next block: `next_input = activations + proj(self_state)`

Training starts with next-token cross-entropy only (projection is zero-initialized, so the model begins as a vanilla GPT). Then an auxiliary objective asks, from `self_state`:

> will my current next-token prediction be correct?

## Models

| Name | Params (approx.) | Self-state | Aux loss weight |
|------|------------------|------------|-----------------|
| `self-state-100m` | ~100M | 128-d | 0.1 |
| `self-state-100m-baseline` | ~100M | off | 0 |

Architecture (both): 12 layers, `n_embd=640`, 10 heads, Pythia-style residuals.

## Config knobs

In `litgpt.Config`:

- `self_state_dim` — set `> 0` to enable (experiment default: `128`)
- `self_state_aux_loss_weight` — BCE weight for correctness prediction (`0` disables)

## Train

```bash
# Self-state + aux correctness loss
litgpt pretrain --config config_hub/pretrain/self-state-100m.yaml

# Matched baseline without self-state
litgpt pretrain --config config_hub/pretrain/self-state-100m-baseline.yaml
```

Phase the objectives if you want a pure CE warmup first:

```bash
# Stage 1: self-state on, aux off
litgpt pretrain --config config_hub/pretrain/self-state-100m.yaml \
  --model_config.self_state_aux_loss_weight 0

# Stage 2: continue with aux on (point --initial_checkpoint_dir at stage 1)
litgpt pretrain --config config_hub/pretrain/self-state-100m.yaml \
  --initial_checkpoint_dir out/pretrain/self-state-100m/final
```

## Experiment question

Does exposing a learned summary of the model's own current state improve its ability to **monitor** (correctness prediction calibration / accuracy) and **control** (downstream LM loss / sample quality) its computation relative to the matched baseline?
