"""顶部相机 (Top Camera) 硬件实现。各实现继承 TopCameraBase。"""

from .base import TopCameraBase
from .opencv import OpenCVTopCamera, OpenCVTopCameraConfig


def make_top_cameras_from_configs(
    configs: dict[str, OpenCVTopCameraConfig],
) -> dict[str, OpenCVTopCamera]:
    """按 {名字: 配置} 构建相机实例字典 (配置驱动)。"""
    return {name: OpenCVTopCamera(cfg, name=name) for name, cfg in configs.items()}


__all__ = [
    "TopCameraBase",
    "OpenCVTopCamera",
    "OpenCVTopCameraConfig",
    "make_top_cameras_from_configs",
]
