"""Use the original gate; never zero condition vectors or replace B14 equations."""
from __future__ import annotations

from contextlib import contextmanager

import torch


def hook_count(unet):
    return len(unet._forward_pre_hooks) + sum(len(unet.down_blocks[i].resnets[0]._forward_pre_hooks) for i in range(4))


@contextmanager
def temporary_cpfm_enabled(unet, cpfm, holder, original_hooks_class, enabled, audit=None, capture_residuals=True):
    """Same original hook routes in both arms. The existing enabled gate is the intervention.

    Caller must remove its idle original B14Hooks first. All temporary hooks are
    removed even if sampling raises. Observation does not alter returned tensors.
    """
    baseline = hook_count(unet)
    physical = [0, 0, 0, 0]
    effective = [0, 0, 0, 0]
    residuals = []

    class ObservedHooks(original_hooks_class):
        def _make_hook(self, index):
            original = super()._make_hook(index)

            def observed(module, inputs):
                physical[index] += 1
                output = original(module, inputs)
                active = bool(self._active(index)) and bool(inputs) and torch.is_tensor(inputs[0]) and holder.get("condition") is not None
                if active:
                    effective[index] += 1
                if audit is not None and capture_residuals and inputs and torch.is_tensor(inputs[0]):
                    x, y = inputs[0], output[0]
                    difference = y.float() - x.float()
                    assert torch.isfinite(difference).all(), "Non-finite injection residual"
                    residuals.append({"layer": index, "physical_call": physical[index], "effective": active,
                                      "residual_mae": float(difference.abs().mean()), "residual_rmse": float(difference.square().mean().sqrt()),
                                      "residual_max_abs": float(difference.abs().max()), "dtype": str(x.dtype), "shape": list(x.shape)})
                return output
            return observed

    holder["enabled"] = bool(enabled)
    hooks = ObservedHooks(unet, cpfm, holder)
    try:
        yield hooks
    finally:
        holder["enabled"] = False
        hooks.close()
        after = hook_count(unet)
        if audit is not None:
            audit.update({"cpfm_injection_enabled": bool(enabled), "physical_prehook_calls_by_layer": physical,
                          "effective_modulations_by_layer": effective, "effective_modulations": sum(effective),
                          "actual_residuals": residuals, "hook_count_before": baseline, "hook_count_after": after,
                          "hook_leak": after != baseline})
        if after != baseline:
            raise RuntimeError("Temporary hook registration leaked")
