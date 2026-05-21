"""
Polyline densification preprocessing.

Splits any segment longer than `max_length` into equal sub-segments so
that the GT road network is well-represented at the anchor resolution.

All coordinates are assumed to be in [-1, 1] normalised image space.
A `max_length` of 0.078 ≈ 40 px in a 512-px image  (40/512 × 2 = 0.156,
but the diagonal of a cell is sqrt(2)*cell_width, so ~0.13 for 8-cell grid).
"""
from __future__ import annotations

import math


def densify_segments(
    segments: list[tuple[float, float, float, float]],
    max_length: float = 0.10,
) -> list[tuple[float, float, float, float]]:
    """Split any segment longer than `max_length` into equal shorter pieces.

    Args:
        segments:   list of (x1, y1, x2, y2) in [-1, 1] coords
        max_length: maximum allowed segment length (same units as coords)

    Returns:
        New list with all segments ≤ max_length.
    """
    out: list[tuple[float, float, float, float]] = []
    for (x1, y1, x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= max_length or length == 0.0:
            out.append((x1, y1, x2, y2))
        else:
            n = math.ceil(length / max_length)
            step = 1.0 / n
            for k in range(n):
                t0, t1 = k * step, (k + 1) * step
                out.append((
                    x1 + t0 * dx, y1 + t0 * dy,
                    x1 + t1 * dx, y1 + t1 * dy,
                ))
    return out
