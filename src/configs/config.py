MODEL_PATH = "models/yolo26m.pt"

# Output
VIDEO_PATH = "data/input/crash.mp4"
LOG_TRACKING_PATH = "data/logs/tracking.csv"
OUTPUT_VIDEO_PATH = "data/output/processed.mp4"
SAVE_MODEL_OUTPUT = True

# Model and detection
CONFIDENCE = 0.2
CLASSES = [2, 3]
MIN_BBOX_AREA = 60 * 60


# Tracking
TRACKER = "bytetrack.yaml"
MAX_FRAME_GAP = 2
MAX_CENTER_DISTANCE = 50
MIN_IOU = 0.3
MAX_SIZE_RATIO = 1.5

# Trajectory and anomaly detection
TRAJECTORY_WINDOW_SIZE = 30
ANOMALY_WINDOW_SIZE = 7
ZSCORE_WINDOW_SIZE = 30
Z_THRESHOLD = 3.0

# Event detection
MERGE_GAP = 4
MIN_CONSECUTIVE_FRAMES = 3