"""Device selection shared by training, evaluation, and replay."""

import torch


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f"Requested {device}, but this CUDA device is unavailable")
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested MPS, but PyTorch cannot access an Apple GPU; use device=auto or cpu")
    elif device.type != "cpu":
        raise ValueError("Supported devices: auto, cuda[:index], mps, cpu")
    return device


def configure_device(cfg):
    device = resolve_device(cfg.get("device", "auto"))
    cfg.device = str(device)
    cfg.compile = bool(cfg.compile and device.type == "cuda")
    if cfg.get("cpu_threads", None):
        torch.set_num_threads(cfg.cpu_threads)
    print(f"Learner device: {device}; compile: {cfg.compile}")
    return device


def mark_step(device):
    if torch.device(device).type == "cuda":
        torch.compiler.cudagraph_mark_step_begin()
