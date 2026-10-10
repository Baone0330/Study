from __future__ import annotations

from pathlib import Path
import sys

from .common import load_original_module, read_json


def install_inference_guards():
    import torch
    import urllib.request
    counters = dict(train_true_calls=0, optimizer_constructions=0, backward_calls=0, download_attempts=0)
    original_train = torch.nn.Module.train

    def train(model, mode=True):
        if mode:
            counters["train_true_calls"] += 1
            raise RuntimeError("Training mode is prohibited")
        return original_train(model, False)

    def prohibit(name):
        def fail(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError("Replay-only operation violated: " + name)
        return fail

    torch.nn.Module.train = train
    torch.optim.Optimizer.__init__ = prohibit("optimizer_constructions")
    torch.Tensor.backward = prohibit("backward_calls")
    torch.autograd.backward = prohibit("backward_calls")
    torch.hub.load_state_dict_from_url = prohibit("download_attempts")
    torch.hub.download_url_to_file = prohibit("download_attempts")
    urllib.request.urlopen = prohibit("download_attempts")
    torch.set_grad_enabled(False)
    return counters


def create_original_model(config, out, guard, device):
    guard.track(config["network_yaml"])
    for path in reversed(config["python_paths"]):
        sys.path.insert(0, path)
    base = load_original_module(config["model_runner"], "historical_native_runner", guard)
    base.OUT = Path(out)
    cfg = base.model_cfg()
    frozen = read_json(guard.track(config["historical_model_config"]))
    if cfg != frozen:
        raise ValueError("Original model_cfg differs from archived resolved historical model config")
    for module in (base, base.formal, base.fixed, base.pre):
        module.OUT = Path(out)
    from ht_prediction_v1.compat import load_original_climatv2_class
    original = load_original_climatv2_class(cfg["climatv2_root"])
    optimizer_factory = original.configure_optimizers
    original.configure_optimizers = lambda self: None
    try:
        model = base.formal.build_model(cfg, device)
    finally:
        original.configure_optimizers = optimizer_factory
    source_root = Path(config["model_root"]).resolve()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file() and Path(path).resolve().is_relative_to(source_root):
            guard.track(path)
    return model, base, cfg
