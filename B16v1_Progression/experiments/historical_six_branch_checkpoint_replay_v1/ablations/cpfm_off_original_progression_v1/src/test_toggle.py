"""Anonymous CPU identity/parity and exception-scope tests using the real original hook."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from cpfm_common import atomic_json, load_module, read_json
from cpfm_toggle_adapter import hook_count, temporary_cpfm_enabled


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnets = nn.ModuleList([nn.Identity()])


class ToyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([Block() for _ in range(4)])

    def forward(self, inputs, timestep):
        return [b.resnets[0](x) for b, x in zip(self.down_blocks, inputs)]


def main(root):
    cfg = read_json(Path(root) / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    original = load_module("toggle_toy_original", cfg["paths"]["cpfm_source"])
    torch.manual_seed(901)
    cpfm = original.B14CPFM().eval().requires_grad_(False)
    cpfm.load_state_dict(torch.load(cfg["paths"]["cpfm_checkpoint"], map_location="cpu", weights_only=False)["model"], strict=True)
    snapshot = {k: v.clone() for k, v in cpfm.state_dict().items()}
    unet = ToyUNet()
    x = [torch.randn(2, c, 8, 8) for c in (320, 320, 640, 1280)]
    condition = torch.randn(16)
    holder = {"enabled": True, "condition": condition, "history_summary": torch.zeros(1, 16), "component_mode": "longitudinal_only", "scheduler_timesteps": [999]}
    old = original.B14Hooks(unet, cpfm, holder)
    with torch.inference_mode():
        on_reference = unet(x, torch.tensor(999))
    old.close()
    assert hook_count(unet) == 0
    with torch.inference_mode():
        with temporary_cpfm_enabled(unet, cpfm, holder, original.B14Hooks, True):
            on = unet(x, torch.tensor(999))
        with temporary_cpfm_enabled(unet, cpfm, holder, original.B14Hooks, False):
            off = unet(x, torch.tensor(999))
    assert all(torch.equal(a, b) for a, b in zip(on_reference, on))
    assert all(torch.equal(a, b) for a, b in zip(x, off))
    assert any(not torch.equal(a, b) for a, b in zip(on, off))
    assert torch.equal(condition, holder["condition"])
    try:
        with temporary_cpfm_enabled(unet, cpfm, holder, original.B14Hooks, True):
            raise ValueError("anonymous intentional exception")
    except ValueError:
        pass
    assert hook_count(unet) == 0 and not holder["enabled"]
    assert all(torch.equal(snapshot[k], v) for k, v in cpfm.state_dict().items())
    qa = {"status": "PASS", "original_on_vs_adapter_bitexact": True, "off_equals_input_bitexact": True,
          "condition_unchanged": True, "weights_unchanged": True, "exception_scope_removes_all_hooks": True,
          "clinical_images_read": 0, "clinical_test_access": 0}
    atomic_json(Path(root) / "pilot/qa/ANONYMOUS_TOGGLE_TEST_V1.json", qa)
    print(json.dumps(qa))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
