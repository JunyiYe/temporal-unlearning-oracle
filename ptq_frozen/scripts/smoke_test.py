from __future__ import annotations

import importlib

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")

    for package in ["numpy", "pandas", "scipy", "sklearn", "yaml", "matplotlib", "tqdm"]:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}={version}")

    import fin_ts_ptq

    print(f"fin_ts_ptq={fin_ts_ptq.__version__}")


if __name__ == "__main__":
    main()
