"""Model-only loader for the ZP-FIT zero-preserving progression release."""
from __future__ import annotations

from pathlib import Path

import torch

from .b14_cpfm import B14CPFM
from .progression_condition import ProgressionCondition
from .reparam_control import ReparamOnly
from .zero_preserving_v1 import B14CPFMZeroPreservingResidual, ZeroPreservingConditionResidual

SEEDS = (42, 3407, 2026)
ARMS = ('ZC', 'ZP', 'ZCP')


def _release_state(path: Path, architecture: str):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'architecture_version', 'state_dict'}:
        raise ValueError('not a minimal release-safe weight payload')
    if payload['architecture_version'] != architecture:
        raise ValueError('architecture version mismatch')
    return payload['state_dict']


def load_zero_preserving_bundle(weight_dir: str | Path, seed: int, arm: str,
                                base_cpfm_weight: str | Path):
    """Return frozen R, P source, condition adapter and CPFM residual models.

    TRUE and ZERO use the same returned models. Call ``cpfm.set_progression(P)``
    for TRUE or ``cpfm.set_progression(torch.zeros_like(P))`` for ZERO.
    The caller owns the frozen SD1.5/LoRA/U-Net runtime and image policy.
    """
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError('unknown formal model seed or arm')
    root = Path(weight_dir)
    R = ReparamOnly()
    R.load_state_dict(_release_state(root / f'R_seed_{seed}.pt', 'ReparamOnly/ZP-FIT-v1'), strict=True)
    P = ProgressionCondition()
    P.load_state_dict(_release_state(root / f'P_source_seed_{seed}.pt', 'ProgressionCondition/ZP-FIT-v1'), strict=True)
    condition = ZeroPreservingConditionResidual()
    cpfm = B14CPFMZeroPreservingResidual()
    base = torch.load(Path(base_cpfm_weight), map_location='cpu', weights_only=True)
    if isinstance(base, dict) and 'state_dict' in base:
        base = base['state_dict']
    if isinstance(base, dict) and 'model_state_dict' in base:
        base = base['model_state_dict']
    if isinstance(base, dict) and 'model' in base:
        base = base['model']
    B14CPFM().load_state_dict(base, strict=True)
    missing = cpfm.load_state_dict(base, strict=False)
    expected = {'a_p', 'progression_projector.layers.0.weight',
                'progression_projector.layers.2.weight'}
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise ValueError('base CPFM key mismatch')
    state = _release_state(root / f'{arm}_seed_{seed}.pt', f'ZeroPreserving{arm}/v1')
    if set(state) != {'condition_adapter', 'cpfm_residual'}:
        raise ValueError('adapter state namespace mismatch')
    condition.load_state_dict(state['condition_adapter'], strict=True)
    cp_state = cpfm.state_dict()
    cp_state.update(state['cpfm_residual'])
    cpfm.load_state_dict(cp_state, strict=True)
    for model in (R, P, condition, cpfm):
        model.eval().requires_grad_(False)
    return R, P, condition, cpfm
