from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional


class PixelCategory(IntEnum):

    INACTIVE = 0
    CHARGE_COLLECTION = 1
    CHARGE_NEIGHBOR = 2
    INDUCTION_ONLY = 3


@dataclass
class PixelClassificationResult:
    """Classification output with pixel IDs for each category.

    Attributes:
        charge_pixels: Pixel IDs for CHARGE_COLLECTION pixels
        neighbor_pixels: Pixel IDs for CHARGE_NEIGHBOR pixels
        induction_pixels: Pixel IDs for INDUCTION_ONLY pixels
        induction_pixels_x: X coordinates (cm) for INDUCTION_ONLY pixels
        induction_pixels_y: Y coordinates (cm) for INDUCTION_ONLY pixels

    Note:
        - "charge_pixels" means pixels in the CHARGE_COLLECTION category
        - The ..._pixels fields contain global pixel IDs (not indices)
        - Types are Optional[Any] to avoid hard dependency on CuPy at import time.
    """

    charge_pixels: Optional[Any]  # (n_charge,) pixel IDs in CHARGE_COLLECTION category
    neighbor_pixels: Optional[Any]  # (n_neighbor,) pixel IDs in CHARGE_NEIGHBOR category
    induction_pixels: Optional[Any]  # (n_induction,) pixel IDs in INDUCTION_ONLY category
    induction_pixels_x: Optional[Any] = None  # (n_induction,) x-coordinates in cm
    induction_pixels_y: Optional[Any] = None  # (n_induction,) y-coordinates in cm

    def summary(self, n_total_pixels: int):
        n_charge = len(self.charge_pixels) if self.charge_pixels is not None else 0
        n_neighbor = len(self.neighbor_pixels) if self.neighbor_pixels is not None else 0
        n_induction = len(self.induction_pixels) if self.induction_pixels is not None else 0
        n_active = n_charge + n_neighbor + n_induction
        lines = [
            "Pixel classification:",
            f"  Total pixels:      {n_total_pixels}",
            f"  Charge collection: {n_charge} ({(100*n_charge/max(n_total_pixels,1)):.2f}%)",
            f"  Charge neighbors:  {n_neighbor} ({(100*n_neighbor/max(n_total_pixels,1)):.2f}%)",
            f"  Induction only:    {n_induction} ({(100*n_induction/max(n_total_pixels,1)):.2f}%)",
            f"  Active total:      {n_active} ({(100*n_active/max(n_total_pixels,1)):.2f}%)",
        ]
        return "\n".join(lines)


from . import voxelization
from . import pixel_classifier

__all__ = [
    'PixelCategory',
    'PixelClassificationResult',
    'voxelization',
    'pixel_classifier',
]
