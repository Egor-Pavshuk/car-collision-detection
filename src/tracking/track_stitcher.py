import math
from .track_state import TrackState


class TrackStitcher:
    def __init__(
        self,
        max_frame_gap: int = 3,
        max_center_distance: float = 120.0,
        min_iou: float = 0.1,
        max_size_ratio: float = 2.0,
    ):
        self.max_frame_gap = max_frame_gap
        self.max_center_distance = max_center_distance
        self.min_iou = min_iou
        self.max_size_ratio = max_size_ratio

        self.next_object_id = 1

        # tracker track_id -> our object_id
        self.track_to_object: dict[int, int] = {}

        # object_id -> latest state
        self.objects: dict[int, TrackState] = {}

    @staticmethod
    def _iou(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        if intersection == 0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _center_distance(
        state: TrackState,
        center_x: float,
        center_y: float,
    ) -> float:
        return math.hypot(
            center_x - state.center_x,
            center_y - state.center_y,
        )

    def _size_similar(
        self,
        state: TrackState,
        width: float,
        height: float,
    ) -> bool:
        if state.width <= 0 or state.height <= 0:
            return False

        width_ratio = max(
            width / state.width,
            state.width / width,
        )

        height_ratio = max(
            height / state.height,
            state.height / height,
        )

        return (
            width_ratio <= self.max_size_ratio
            and height_ratio <= self.max_size_ratio
        )

    def _find_candidate(
        self,
        frame: int,
        bbox: tuple[float, float, float, float],
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        track_id: int,
    ) -> int | None:

        best_object_id = None
        best_score = float("-inf")

        for object_id, state in self.objects.items():

            if state.track_id == track_id:
                continue

            frame_gap = frame - state.frame

            if frame_gap < 0 or frame_gap > self.max_frame_gap:
                continue

            distance = self._center_distance(
                state,
                center_x,
                center_y,
            )

            iou = self._iou(
                state.bbox,
                bbox,
            )

            size_similar = self._size_similar(
                state,
                width,
                height,
            )

            if not size_similar:
                continue

            if distance > self.max_center_distance:
                continue

            if frame_gap > 0 and iou < self.min_iou:
                continue

            score = (
                -distance
                + iou * self.max_center_distance
                - frame_gap * 20.0
            )

            if score > best_score:
                best_score = score
                best_object_id = object_id

        return best_object_id

    def update(
        self,
        frame: int,
        track_id: int,
        bbox: tuple[float, float, float, float],
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> tuple[int, bool]:

        if track_id in self.track_to_object:
            object_id = self.track_to_object[track_id]

            self.objects[object_id] = TrackState(
                object_id=object_id,
                track_id=track_id,
                frame=frame,
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
            )

            return object_id, False

        object_id = self._find_candidate(
            frame=frame,
            bbox=bbox,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            track_id=track_id,
        )

        stitched = object_id is not None

        if object_id is None:
            object_id = self.next_object_id
            self.next_object_id += 1

        self.track_to_object[track_id] = object_id

        self.objects[object_id] = TrackState(
            object_id=object_id,
            track_id=track_id,
            frame=frame,
            bbox=bbox,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
        )

        return object_id, stitched