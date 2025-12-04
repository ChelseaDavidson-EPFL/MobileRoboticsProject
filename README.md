# Mobile Robotics Project - Group 40

**Autonomous Navigation System for Thymio Robot**

## Team Members
- Arthur Bauer
- Chelsea Davidson
- Eleonore Salin
- Quitterie Vermeulen

## Project Overview

This project implements an autonomous navigation system for the Thymio II robot, integrating computer vision, path planning, sensor fusion, and motion control to enable robust navigation in a known environment with obstacle avoidance capabilities.

### Key Features
- **Vision-based localization** using ArUco markers
- **Hybrid navigation** combining global A* path planning with reactive local obstacle avoidance
- **Extended Kalman Filter** for sensor fusion and state estimation
- **Astolfi controller** for smooth waypoint tracking
- **Mode switching** between GLOBAL, LOCAL, and KIDNAPPED states

## System Architecture

The system consists of six modular components:

1. **Vision Module** (`vision.py`) - Camera-based localization and environment mapping
2. **Global Navigation** (`global_nav.py`) - A*/Dijkstra path planning on occupancy grid
3. **Local Navigation** (`local_nav.py`) - Reactive obstacle avoidance using IR sensors
4. **Filtering** (`filtering.py`) - Extended Kalman Filter for state estimation
5. **Motion Control** (`motion_controll.py`) - Astolfi controller for waypoint tracking
6. **Thymio Interface** (`thymio.py`) - Hardware abstraction layer
7. **Utilities** (`utils.py`) - Constants and coordinate transformations

## Hardware Requirements

- **Robot:** Thymio II (11 cm × 11 cm × 5 cm)
  - 7 horizontal IR proximity sensors (5 front, 2 rear)
  - 2 ground IR sensors
  - Differential drive (wheel distance: 9.5 cm)
- **Camera:** External webcam for environment and robot detection
- **Connection:** USB cable (recommended over wireless for lower latency)
- **Arena:** Square workspace with white background
- **Markers:** ArUco markers (IDs: 0=robot, 1,2=arena corners, 3=goal)
- **Obstacles:** Red paper cutouts

## Software Dependencies

```bash
pip install numpy opencv-python matplotlib tdmclient asyncio
```

### Required Libraries
- `numpy` - Array operations and linear algebra
- `opencv-python` (cv2) - Computer vision and ArUco marker detection
- `matplotlib` - Path visualization
- `tdmclient` - Thymio robot interface
- `asyncio` - Asynchronous sensor communication

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChelseaDavidson-EPFL/MobileRoboticsProject.git
   cd MobileRoboticsProject
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Connect Thymio:**
   - Connect Thymio II via USB cable
   - Ensure tdmclient can detect the robot

4. **Setup environment:**
   - Print ArUco markers (IDs 0, 1, 2, 3)
   - Place markers on white background for arena
   - Attach marker ID 0 to top of Thymio (covering entire surface)
   - Cut red paper obstacles

## Usage

### Running the Full System

Open `Group_40_report.ipynb` in Jupyter and execute the main loop cell, or run:

```python
# See Group_40_report.ipynb cell 4 for complete implementation
```

The main loop executes at **10 Hz** and integrates all modules for autonomous navigation.

### Testing Individual Modules

The Jupyter notebook contains executable examples for each module:

- **Vision:** Cell 12 - Environment detection and robot pose tracking
- **Global Navigation:** Cell 15 - Path planning with test grid
- **Local Navigation:** Cell 17 - Obstacle avoidance (requires robot)
- **Filtering:** Cells 20-28 - EKF and variance experiments
- **Motion Control:** Cell 30 - Waypoint tracking (requires robot)
- **Utils:** Cell 10 - Coordinate transformations

## Configuration

Key parameters in `utils.py`:

```python
GRID_DIM = 200              # Grid resolution (200×200 cells)
FREQ_MAIN_LOOP = 10         # Control loop frequency (Hz)
SAFETY_MARGIN = 10          # Obstacle expansion margin (cells)
```

Filter noise parameters (experimentally determined):
- Process noise (Q): Position, orientation, velocity uncertainties
- Measurement noise (R): Camera measurement uncertainties

## Project Structure

```
MobileRoboticsProject/
├── Group_40_report.ipynb    # Main notebook with documentation and code
├── vision.py                # Computer vision module
├── global_nav.py           # Path planning algorithms
├── local_nav.py            # Obstacle avoidance logic
├── filtering.py            # Extended Kalman Filter
├── motion_controll.py      # Motion control (Astolfi)
├── thymio.py               # Robot interface
├── utils.py                # Constants and utilities
├── pictures/               # Images for documentation
│   ├── environment.jpg
│   ├── grid.png
│   └── ...
└── README.md              # This file
```

## Documentation

Complete documentation is available in `Group_40_report.ipynb`, including:
- Detailed theory for each module
- Design choices and rationale
- Parameter tuning methodology
- Integration examples
- Experimental results

## Navigation Modes

The system operates in three modes:

| Mode | Description | Trigger |
|------|-------------|---------|
| **GLOBAL** | Follows planned path | Default mode |
| **LOCAL** | Reactive obstacle avoidance | IR sensors detect obstacle |
| **KIDNAPPED** | Robot lifted, awaits re-localization | Ground sensors detect lift |

## Known Limitations

- **External camera dependency:** Workspace limited to camera field of view
- **ArUco marker requirement:** Markers must remain visible for localization
- **Low execution frequency (10 Hz):** Limits maximum safe speed
- **Wired connection required:** Wireless introduces latency issues
- **Virtual obstacles:** IR sensors detect physical objects not in initial map
- **External computation:** All processing done on laptop, not onboard

## References

See `Group_40_report.ipynb` References section for complete citations.

Key references:
- OpenCV Documentation - ArUco markers, homography, color segmentation
- Welch & Bishop - "An Introduction to the Kalman Filter" (1995)
- EPFL Basics of Mobile Robotics Course (2024)

## License

This project was developed as part of the Basics of Mobile Robotics course at EPFL (Fall 2024).

## Contact

For questions or issues, please contact the team members through the course platform.

---

**Note:** This project requires a physical Thymio II robot and camera setup for complete execution. Full simulation mode is not available. Individual modules can be tested without full setup.
