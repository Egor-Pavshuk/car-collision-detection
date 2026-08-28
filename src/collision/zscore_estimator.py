from collections import deque
import numpy as np

SMOOTH_VALUE = 1e-6

class ZScoreEstimator:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size

        self.score_history: dict[int, deque[float]] = {}

    def update(
        self,
        object_id: int,
        score: float | None,
    ) -> float | None:

        if score is None:
            return None

        if object_id not in self.score_history:
            self.score_history[object_id] = deque(
                maxlen=self.window_size
            )

        history = self.score_history[object_id]

        if len(history) < 2:
            history.append(score)
            return None

        mean = np.mean(history)
        std = max(np.std(history), SMOOTH_VALUE)

        zindex = abs(score - mean) / std

        history.append(score)

        return float(zindex)