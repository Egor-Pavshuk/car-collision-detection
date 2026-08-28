from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrackState:
    object_id: int
    track_id: int
    frame: int

    bbox: Tuple[float, float, float, float]

    center_x: float
    center_y: float

    width: float
    height: float