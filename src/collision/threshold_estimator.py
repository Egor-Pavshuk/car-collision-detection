from collections import deque
import numpy as np

MIN_COUNT_OF_SCORES = 5

class ThresholdEstimator:
    def __init__(self, percentile: float = 0.95, window_size: int = 30):
        self.percentile = percentile
        self.window_size = window_size
        self.scores: dict[int, deque[float]] = {}

    def update(
        self,
        object_id: int,
        score: float | None,
    ) -> float | None:

        if score is None:
            return None

        if object_id not in self.scores:
            self.scores[object_id] = deque(maxlen=self.window_size)

        history = self.scores[object_id]

        if len(history) < MIN_COUNT_OF_SCORES:
            history.append(score)

            if len(history) < MIN_COUNT_OF_SCORES:
                return None

            return float(
                np.quantile(history, self.percentile)
            )

        threshold = float(
            np.quantile(history, self.percentile)
        )

        # if score <= threshold:
        #     history.append(score)

        history.append(score)

        return float(
            np.quantile(history, self.percentile)
        )