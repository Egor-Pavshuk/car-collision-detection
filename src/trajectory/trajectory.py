from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    frame: int
    center_x: float
    center_y: float


class TrajectoryTracker:
    def __init__(self, window_size: int = 10, ):
        self.window_size = window_size

        self.history: dict[int, deque[TrajectoryPoint]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(
        self,
        object_id: int,
        frame: int,
        center_x: float,
        center_y: float,
    ) -> dict:

        history = self.history[object_id]

        dx = None
        dy = None

        if history:
            previous = history[-1]

            frame_diff = frame - previous.frame

            if frame_diff > 0:
                dx = (center_x - previous.center_x) / frame_diff
                dy = (center_y - previous.center_y) / frame_diff

        history.append(
            TrajectoryPoint(
                frame=frame,
                center_x=center_x,
                center_y=center_y,
            )
        )

        return {
            "dx": dx,
            "dy": dy,
        }