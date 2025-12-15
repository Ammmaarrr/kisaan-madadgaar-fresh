import importlib
pkgs = ["matplotlib", "seaborn", "shap", "lime", "timm", "skimage", "torch", "torchvision"]
for name in pkgs:
    module_name = "skimage" if name == "skimage" else name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "available")
        print(f"{name}: {version}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: MISSING ({exc})")
