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
    mixed_precision = bool(cfg.get("mixed_precision", False) and device.type == "cuda")
    if mixed_precision:
        with torch.cuda.device(device):
            mixed_precision = torch.cuda.is_bf16_supported(including_emulation=False)
    cfg.mixed_precision = mixed_precision
    cfg.channels_last = bool(cfg.get("channels_last", False) and cfg.obs == "rgb"
                             and device.type == "cuda")
    cfg.fused_optimizer = bool(cfg.get("fused_optimizer", False) and device.type == "cuda")
    if cfg.get("cpu_threads", None):
        torch.set_num_threads(cfg.cpu_threads)
    precision = "BF16 mixed precision" if cfg.mixed_precision else "FP32"
    print(f"Learner device: {device}; compile: {cfg.compile}; "
          f"updates: {precision}; channels_last: {cfg.channels_last}; "
          f"fused_optimizer: {cfg.fused_optimizer}")
    return device


def mark_step(device):
    if torch.device(device).type == "cuda":
        torch.compiler.cudagraph_mark_step_begin()
