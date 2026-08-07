from .base import Generator, GuidanceHook


def get_generator(name: str, device: str = "cpu", **kw) -> Generator:
    if name == "pixel":
        from .pixel import PixelGenerator
        return PixelGenerator(device=device, **kw)
    if name == "sdxl":
        from .sdxl import SDXLGenerator
        return SDXLGenerator(device=device, **kw)
    if name == "deepfloyd":
        from .deepfloyd import DeepFloydGenerator
        return DeepFloydGenerator(device=device, **kw)
    raise KeyError(f"unknown generator {name!r}; have pixel|sdxl|deepfloyd")


__all__ = ["Generator", "GuidanceHook", "get_generator"]
