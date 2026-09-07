# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
import torch.nn.functional as F

from litgpt import GPT, Config
from litgpt.model import SelfStateCell
from litgpt.utils import chunked_cross_entropy, num_parameters


def test_self_state_config_defaults() -> None:
    config = Config(block_size=32, n_layer=2, n_embd=32, n_head=4, padded_vocab_size=128)
    assert config.self_state_dim == 0
    assert config.self_state_aux_loss_weight == 0.0


def test_self_state_100m_named_configs() -> None:
    cfg = Config.from_name("self-state-100m")
    assert cfg.self_state_dim == 128
    assert cfg.self_state_aux_loss_weight == 0.1
    assert cfg.n_layer == 12
    assert cfg.n_embd == 640

    baseline = Config.from_name("self-state-100m-baseline")
    assert baseline.self_state_dim == 0
    assert baseline.n_embd == cfg.n_embd
    assert baseline.n_layer == cfg.n_layer


def test_self_state_cell_gate_and_shapes() -> None:
    cell = SelfStateCell(n_embd=32, state_dim=8)
    activations = torch.randn(2, 5, 32)
    state = torch.zeros(2, 5, 8)
    new_state, delta = cell(activations, state)
    assert new_state.shape == (2, 5, 8)
    assert delta.shape == (2, 5, 32)
    # Zero-init projection should contribute nothing at construction time.
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_gpt_self_state_forward_and_aux_loss() -> None:
    config = Config(
        block_size=16,
        n_layer=2,
        n_embd=32,
        n_head=4,
        padded_vocab_size=64,
        self_state_dim=8,
        self_state_aux_loss_weight=0.5,
    )
    model = GPT(config)
    assert model.self_state_cells is not None
    assert model.correctness_head is not None
    assert len(model.self_state_cells) == config.n_layer

    idx = torch.randint(0, config.padded_vocab_size, (2, 16))
    targets = torch.randint(0, config.padded_vocab_size, (2, 16))

    logits = model(idx)
    assert logits.shape == (2, 16, config.padded_vocab_size)

    logits, self_state = model(idx, return_self_state=True)
    assert self_state.shape == (2, 16, config.self_state_dim)

    ce_loss = chunked_cross_entropy(logits, targets)
    correct = (logits.argmax(dim=-1) == targets).float()
    aux_loss = F.binary_cross_entropy_with_logits(model.correctness_head(self_state).squeeze(-1), correct)
    loss = ce_loss + config.self_state_aux_loss_weight * aux_loss
    loss.backward()

    grads = [p.grad for p in model.self_state_cells.parameters() if p.grad is not None]
    assert grads, "self-state parameters should receive gradients from the joint loss"


def test_self_state_disabled_matches_vanilla_modules() -> None:
    config = Config(block_size=16, n_layer=2, n_embd=32, n_head=4, padded_vocab_size=64)
    model = GPT(config)
    assert model.self_state_cells is None
    assert model.correctness_head is None


def test_self_state_100m_parameter_ballpark() -> None:
    model = GPT(Config.from_name("self-state-100m"))
    n_params = num_parameters(model)
    # Roughly 100M; allow a wide band for vocab/head accounting.
    assert 70_000_000 < n_params < 160_000_000, n_params
