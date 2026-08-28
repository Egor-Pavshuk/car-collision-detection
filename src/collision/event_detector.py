from dataclasses import dataclass


@dataclass
class CollisionEvent:
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_score: float
    peak_threshold: float


class EventDetector:
    def __init__(
        self,
        merge_gap: int = 10,
        min_consecutive_frames: int = 3
    ):
        self.merge_gap = merge_gap
        self.min_consecutive_frames = min_consecutive_frames

        self.current_events: dict[int, CollisionEvent] = {}
        self.events: dict[int, list[CollisionEvent]] = {}

        self.consecutive_counts: dict[int, int] = {}

    def update(
        self,
        frame: int,
        object_id: int,
        score: float | None,
        threshold: float | None
    ) -> CollisionEvent | None:

        if score is None or threshold is None:
            self.consecutive_counts[object_id] = 0
            return None

        if score < threshold:
            self.consecutive_counts[object_id] = 0
            current_event = self.current_events.get(object_id)

            if current_event is not None:

                gap = frame - current_event.end_frame

                if gap > self.merge_gap:

                    finished = current_event

                    self.events.setdefault(object_id, []).append(finished)

                    del self.current_events[object_id]

                    return finished

            return None

        self.consecutive_counts[object_id] = (self.consecutive_counts.get(object_id, 0) + 1)

        current_event = self.current_events.get(object_id)

        # Start a new event
        if current_event is None:
            if (self.consecutive_counts[object_id] < self.min_consecutive_frames):
                return None
            
            self.current_events[object_id] = CollisionEvent(
                start_frame=frame - self.min_consecutive_frames + 1,
                end_frame=frame,
                peak_frame=frame,
                peak_score=score,
                peak_threshold=threshold,
            )

            return None

        gap = frame - current_event.end_frame

        # Continue existing event
        if gap <= self.merge_gap:
            current_event.end_frame = frame

            if score > current_event.peak_score:
                current_event.peak_score = score
                current_event.peak_frame = frame
                current_event.peak_threshold = threshold

            return None

        # If gap is too large, close existing event
        finished = current_event

        self.events.setdefault(object_id, []).append(finished)
        self.consecutive_counts[object_id] = 1
        del self.current_events[object_id]

        return finished

    def close_expired(self, frame: int) -> list[CollisionEvent]:

        finished_events = []

        for object_id, event in list(self.current_events.items()):

            gap = frame - event.end_frame

            if gap > self.merge_gap:

                self.events.setdefault(object_id, []).append(event)

                finished_events.append(event)

                del self.current_events[object_id]

        return finished_events

    def finalize(self) -> list[CollisionEvent]:

        finished_events = []

        for object_id, event in self.current_events.items():
            self.events.setdefault(object_id, []).append(event)
            finished_events.append(event)

        self.current_events.clear()

        return finished_events