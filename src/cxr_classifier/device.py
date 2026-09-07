"""Device detection helper.

Picks the best available compute device in this order:
1. CUDA (NVIDIA GPUs)
2. DirectML (AMD/Intel GPUs on Windows, via torch-directml)
3. MPS (Apple Silicon)
4. CPU (fallback)

Returns a string suitable for `torch.device(...)` *except* for the DirectML
case, which requires `torch_directml.device()`. Callers should use
:func:`get_torch_device` which handles that subtlety.
"""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import torch_directml  # type: ignore

    _HAS_DIRECTML = True
except ImportError:
    _HAS_DIRECTML = False


# Map of user-facing alias -> torch device string
DEVICE_ALIASES = {
    "auto": None,  # special: pick best
    "cpu": "cpu",
    "cuda": "cuda",
    "gpu": None,  # alias for auto (CUDA → DirectML → MPS → CPU)
    "directml": "directml",
    "mps": "mps",
}


def detect_best_device() -> Tuple[torch.device, str]:
    """Return ``(device, name)`` for the best available accelerator.

    ``name`` is a human-readable string ("cuda", "directml", "mps", "cpu").
    The returned :class:`torch.device` is constructed correctly for each
    backend, including DirectML's ``privateuseone`` naming.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    if _HAS_DIRECTML:
        try:
            dml = torch_directml.device()
            # torch_directml.device() returns a torch.device("privateuseone:0")
            # under the hood. We expose "directml" as the friendly name.
            return dml, "directml"
        except Exception:
            pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"

    return torch.device("cpu"), "cpu"


def get_torch_device(device_arg: str) -> torch.device:
    """Resolve a user-supplied device string to a :class:`torch.device`.

    Accepts: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"gpu"``, ``"directml"``,
    ``"mps"``, or any standard :class:`torch.device` string.
    """
    if device_arg in ("auto", "gpu", None):
        dev, _ = detect_best_device()
        return dev

    if device_arg == "directml":
        if not _HAS_DIRECTML:
            raise RuntimeError(
                "DirectML requested but torch-directml is not installed. "
                "Run: pip install torch-directml"
            )
        return torch_directml.device()

    if device_arg in DEVICE_ALIASES and DEVICE_ALIASES[device_arg] is not None:
        return torch.device(DEVICE_ALIASES[device_arg])

    # Last resort: let torch try to parse it
    return torch.device(device_arg)


def device_kind(device: torch.device) -> str:
    """Return the backend kind ("cuda", "directml", "mps", "cpu") for a device."""
    if device.type == "cuda":
        return "cuda"
    if _HAS_DIRECTML and device.type == "privateuseone":
        return "directml"
    return device.type


def supports_amp(device: str) -> bool:
    """Whether the given device kind supports automatic mixed precision."""
    # DirectML supports AMP via torch.amp.autocast(device_type="privateuseone").
    # CPU/MPS have no native AMP for inference; AMP on MPS is also unstable.
    return device in ("cuda", "directml")