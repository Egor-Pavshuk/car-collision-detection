import csv, cv2
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from src.collision.zscore_estimator import ZScoreEstimator
from src.configs.config import MODEL_PATH,  MIN_BBOX_AREA, CONFIDENCE, CLASSES
from src.configs.config import SAVE_MODEL_OUTPUT, VIDEO_PATH, LOG_TRACKING_PATH, OUTPUT_VIDEO_PATH
from src.configs.config import TRACKER, MAX_FRAME_GAP, MAX_CENTER_DISTANCE, MIN_IOU, MAX_SIZE_RATIO
from src.configs.config import TRAJECTORY_WINDOW_SIZE, ANOMALY_WINDOW_SIZE, ZSCORE_WINDOW_SIZE, Z_THRESHOLD
from src.configs.config import MERGE_GAP, MIN_CONSECUTIVE_FRAMES
from src.tracking.track_stitcher import TrackStitcher
from src.trajectory.trajectory import TrajectoryTracker
from src.collision.anomaly_score import AnomalyScore
from src.collision.event_detector import EventDetector
from src.collision.threshold_estimator import ThresholdEstimator
from src.visualization.render_state import RenderObject, RenderState
from src.visualization.renderer import Renderer

LOG_TRACKING_PATH = Path(LOG_TRACKING_PATH)
OUTPUT_VIDEO_PATH = Path(OUTPUT_VIDEO_PATH)

def main():

    model = YOLO(MODEL_PATH)

    LOG_TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    results = model.track(
        source=VIDEO_PATH,
        device=0,
        conf=CONFIDENCE,
        classes=CLASSES,
        tracker=TRACKER,
        save=SAVE_MODEL_OUTPUT,
        project="data/output",
        name="baseline",
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    stitcher = TrackStitcher(
        max_frame_gap=MAX_FRAME_GAP,
        max_center_distance=MAX_CENTER_DISTANCE,
        min_iou=MIN_IOU,
        max_size_ratio=MAX_SIZE_RATIO,
    )

    trajectory = TrajectoryTracker(
        window_size=TRAJECTORY_WINDOW_SIZE,
    )

    anomaly = AnomalyScore(
        window_size=ANOMALY_WINDOW_SIZE,
    )

    event_detector = EventDetector(
        merge_gap=MERGE_GAP,
        min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
    )

    threshold_estimator = ThresholdEstimator(
        percentile=0.95,
        window_size=30,
    )

    zscore_estimator = ZScoreEstimator(
        window_size=ZSCORE_WINDOW_SIZE
    )

    renderer = Renderer()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (width, height),
    )

    if not video_writer.isOpened():
        raise RuntimeError(
            f"Failed to open video writer: {OUTPUT_VIDEO_PATH}"
        )

    with open(LOG_TRACKING_PATH, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "track_id",
            "object_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "dx",
            "dy",
            "anomaly_score",
            "zscore",
            "stitched",
        ])

        for frame_idx, result in enumerate(tqdm(results, total=total_frames, desc="Processing video", unit="frame")):

            boxes = result.boxes

            if boxes is None:
                continue

            render_objects: list[RenderObject] = []

            frame_object_ids = set()

            for box in boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                width = x2 - x1
                height = y2 - y1

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                bbox = (
                    x1,
                    y1,
                    x2,
                    y2,
                )

                # Min box area constraint
                if width * height <= MIN_BBOX_AREA:
                    continue

                # -----------------------------
                # TRACK STITCHING
                # -----------------------------

                object_id, stitched = stitcher.update(
                    frame=frame_idx,
                    track_id=track_id,
                    bbox=bbox,
                    center_x=center_x,
                    center_y=center_y,
                    width=width,
                    height=height,
                )

                # -----------------------------
                # TRAJECTORY
                # -----------------------------

                features = trajectory.update(
                    object_id=object_id,
                    frame=frame_idx,
                    center_x=center_x,
                    center_y=center_y,
                )

                dx = features["dx"]
                dy = features["dy"]

                trajectory_points = [
                    (point.center_x, point.center_y)
                    for point in trajectory.history[object_id]
                ]

                # -----------------------------
                # ANOMALY
                # -----------------------------

                score = anomaly.update(
                    object_id=object_id,
                    dx=dx,
                )

                zscore = zscore_estimator.update(object_id=object_id, score=score)

                # threshold = threshold_estimator.update(object_id=object_id, score=score)
                threshold = Z_THRESHOLD

                if score is not None:
                
                    event_detector.update(
                        frame=frame_idx,
                        object_id=object_id,
                        threshold=threshold,
                        score=zscore,
                    )

                is_event = object_id in event_detector.current_events
                is_anomaly = (
                    score is not None
                    and threshold is not None
                    and score >= threshold
                )

                frame_object_ids.add(object_id)
                class_name = model.names[class_id]

                render_objects.append(
                    RenderObject(object_id=object_id, 
                                 track_id=track_id, 
                                 class_name=class_name,
                                 conf=confidence, 
                                 bbox=bbox,
                                 score=zscore, #score,
                                 threshold=threshold,
                                 is_anomaly=is_anomaly,
                                 is_event=is_event,
                                 trajectory=trajectory_points)
                    )

                # -----------------------------
                # LOG
                # -----------------------------

                writer.writerow([
                    frame_idx,
                    track_id,
                    object_id,
                    class_id,
                    class_name,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                    center_x,
                    center_y,
                    dx,
                    dy,
                    score,
                    zscore,
                    stitched,
                ])

            event_detector.close_expired(frame_idx)

            inference_time_ms = result.speed["inference"]
            vehicles_count = len(render_objects)
            render_state = RenderState(frame=frame_idx, inference_time_ms=inference_time_ms, objects=render_objects, vehicles_count=vehicles_count)
            
            frame = renderer.render(result.orig_img.copy(), render_state)
            video_writer.write(frame)

    video_writer.release()
    remaining_events = event_detector.finalize()

    print("\nDetected collision events:")

    for object_id, events in event_detector.events.items():
        for event in events:
            ratio = event.peak_score / event.peak_threshold
            print(
                f"object={object_id}: "
                f"{event.start_frame} -> {event.end_frame}, "
                f"peak={event.peak_frame}, "
                f"score={event.peak_score:.2f} "
                f"threshold={event.peak_threshold:.2f}"
            )


if __name__ == "__main__":
    main()