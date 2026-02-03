"""
System Resource Monitor.

This module provides utilities to check system resources such as RAM
usage and disk space, assisting in monitoring during resource-intensive tasks.
"""

import shutil
from typing import Tuple

import psutil


class ResourceMonitor:
    """Monitors system resources (RAM, Disk)."""

    @staticmethod
    def get_ram_usage() -> Tuple[float, float, float]:
        """Returns (used_gb, total_gb, percent)."""
        try:
            mem = psutil.virtual_memory()
            return (mem.used / (1024**3), mem.total / (1024**3), mem.percent)
        except ImportError:
            return (0.0, 0.0, 0.0)

    @staticmethod
    def get_disk_free(path: str = ".") -> float:
        """Returns free space in GB."""
        _, _, free = shutil.disk_usage(path)
        return free / (1024**3)

    @staticmethod
    def get_resource_status() -> str:
        """Returns formatted string for UI."""
        used, total, percent = ResourceMonitor.get_ram_usage()
        disk_free = ResourceMonitor.get_disk_free()

        # Fallback if psutil fails (Mac native)
        if total == 0:
            try:
                # Simplified for brevity, ideally would use full logic from system_utils
                return f"Disk: {disk_free:.1f} GB free"
            except Exception:
                return "N/A"

        return f"RAM: {used:.1f}/{total:.1f} GB ({percent:.1f}%) | Disk: {disk_free:.1f} GB free"

    @staticmethod
    def estimate_model_size(params_billions: float, quantization_bits: int = 4) -> float:
        """Estimates model size in GB."""
        # 1B params @ 16-bit = 2GB
        # 1B params @ 4-bit = 0.5GB + overhead
        size_gb = params_billions * (quantization_bits / 8.0)
        return size_gb * 1.2  # 20% overhead
