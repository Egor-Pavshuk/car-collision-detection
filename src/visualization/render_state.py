from dataclasses import dataclass

@dataclass
class RenderObject:
    object_id: int
    track_id: int
    class_name: str

    conf: float | None

    bbox: tuple[float, float, float, float]

    score: float | None
    threshold: float | None

    is_anomaly: bool
    is_event: bool

    trajectory: list[tuple[float, float]]

@dataclass
class RenderState:
    frame: int
    inference_time_ms: float
    vehicles_count: int
    objects: list[RenderObject]

