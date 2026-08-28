from collections import deque
import numpy as np


class AnomalyScore:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size

        self.dx_history: dict[int, deque[float]] = {}

    def update(
        self,
        object_id: int,
        dx: float | None,
    ) -> float | None:

        if dx is None:
            return None

        if object_id not in self.dx_history:
            self.dx_history[object_id] = deque(
                maxlen=self.window_size
            )

        history = self.dx_history[object_id]

        if len(history) < 2:
            history.append(dx)
            return None

        rolling_mean = np.mean(history)

        score = abs(dx - rolling_mean)

        history.append(dx)

        return float(score)