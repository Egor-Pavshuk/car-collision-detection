import cv2
from numpy import ndarray
from .render_state import RenderState

class Renderer:

    def render(self, frame: ndarray, state: RenderState) -> ndarray:
        for obj in state.objects:

            x1, y1, x2, y2 = map(int, obj.bbox)

            color = (0, 255, 0)

            if obj.is_anomaly:
                color = (0, 255, 255)

            if obj.is_event:
                color = (0, 0, 255)

            points = [
                (int(x), int(y))
                for x, y in obj.trajectory
            ]

            for i in range(1, len(points)):
                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    color,
                    2,
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = f"ID: {obj.object_id}"

            if obj.class_name is not None:
                label += f" {obj.class_name}"

            if obj.score is not None:
                label += f" score={obj.score:.2f}"

            if obj.threshold is not None:
                label += f" th={obj.threshold:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )

        # Inference time & Vehicles count
        cv2.putText(
            frame,
            f"Inference: {state.inference_time_ms:.1f} ms",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 40, 40),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Pipeline: {state.pipeline_time_ms} ms",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 40, 40),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Vehicles count: {state.vehicles_count}",
            (30, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 40, 40),
            3,
            cv2.LINE_AA,
        )

        return frame