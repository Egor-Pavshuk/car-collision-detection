# Vehicle Collision Detection

## Overview

This investigation was conducted to detect vehicle collisions in traffic video. The implemented pipeline is based on detecting anomalous vehicle behavior. Due to unstable detection of the second vehicle involved in the collision, several approaches were investigated to identify the collision from the behavior of a tracked vehicle.

## Approach
1. YOLO detection
2. ByteTrack
3. Track stitching
4. Trajectory extraction
5. Anomaly score
6. Z-score normalization
7. Event detection

## Pipeline
- run YOLO tracking inference in stream mode
- detect vehicles
- track vehicles using ByteTrack
- stitch track IDs into persistent object IDs
- calculate horizontal displacement `dx` (px/frame) between consecutive frames
- calculate the absolute deviation of `dx` from its rolling mean as the anomaly score, using a window size of 7
- calculate the Z-score of the anomaly score using a window size of 30
- detect collision candidates using a Z-score threshold of 3.0 and temporal filtering
- log extracted features
- render processed video frames with detected vehicles, trajectories, and anomaly information

## Project Structure

```
car-collision-detection
├── data
│   ├── input/
│   ├── logs/
│   └── output/
├── models/
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── collision/
│   │   ├── anomaly_score.py
│   │   ├── event_detector.py
│   │   ├── threshold_estimator.py
│   │   └── zscore_estimator.py
│   ├── configs/
│   │   └── config.py
│   ├── tracking/
│   │   ├── track_state.py
│   │   └── track_stitcher.py
│   ├── trajectory/
│   │   └── trajectory.py
│   ├── visualization/
│   │   ├── render_state.py
│   │   └── renderer.py
│   └── main.py
├── tests/
├── .gitignore
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Installation

## Usage

## Configuration

## Results

## Limitations

## Possible Improvements