"""One-line description of the device and software stack, printed at the top of benchmarks."""

import platform
import subprocess

import torch


def _driver_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"


def describe(device: torch.device | str = "cuda") -> str:
    """Return e.g. ``gpu=NVIDIA A40 sm_86 84SMs driver=575.57.08 torch=2.8.0+cu128 cuda=12.8``."""
    device = torch.device(device)
    software = (
        f"torch={torch.__version__} cuda={torch.version.cuda} python={platform.python_version()}"
    )
    if device.type != "cuda":
        return f"device={device} cpu={platform.processor() or platform.machine()} {software}"
    props = torch.cuda.get_device_properties(device)
    return (
        f"gpu={props.name} sm_{props.major}{props.minor} {props.multi_processor_count}SMs "
        f"driver={_driver_version()} {software}"
    )
