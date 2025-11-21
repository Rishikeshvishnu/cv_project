# 3D Scene Reconstruction using Structure from Motion (SfM)

A Python implementation of a Structure from Motion pipeline that reconstructs 3D scene geometry and camera poses from multiple 2D images.

## Overview

This project implements a complete Structure from Motion pipeline that takes a sequence of images and reconstructs:
- 3D point cloud of the scene
- Camera poses (rotation and translation) for each image
- Visual representation of the reconstructed cameras and 3D points

The implementation uses classical computer vision techniques including SIFT feature matching, essential matrix estimation, and robust estimation with MSAC (M-estimator Sample Consensus).

## Features

- **SIFT Feature Detection**: Robust feature extraction and matching across images
- **Essential Matrix Estimation**: Multiple methods including 5-point, 8-point, and homography-based approaches
- **Robust Estimation**: MSAC algorithm for outlier rejection
- **Camera Pose Estimation**: Computes rotation and translation for each camera
- **3D Triangulation**: Direct Linear Transform (DLT) for 3D point reconstruction
- **Non-linear Refinement**: Levenberg-Marquardt style optimization for camera parameters
- **Multiple Datasets**: Pre-configured support for 11 different datasets

## Requirements

```
numpy
scipy
opencv-python (cv2)
matplotlib
```

Install dependencies:
```bash
pip install numpy scipy opencv-python matplotlib
```

## Usage

Run the reconstruction on a specific dataset:

```bash
python main.py <dataset_id>
```

Where `<dataset_id>` is a number from 1-11 corresponding to the available datasets.

### Example

```bash
python main.py 1
```

This will process the kronan dataset and display the 3D reconstruction.

## Datasets

The project includes 11 pre-configured datasets:

1. **Kronan** - 2 images (1936×1296)
2. **Courtyard Corner** - 9 images (1936×1296)
3. **Cathedral Gate** - 12 images (1936×1296)
4. **Fountain** - 14 images (1936×1296)
5. **Golden Statue** - 10 images (1936×1296)
6. **Landhaus Graz Detail** - 8 images (2272×1704)
7. **Heidelberg Building** - 7 images (2272×1704)
8. **Relief** - 11 images (2272×1704)
9. **Triceratops Poster** - 9 images (2272×1704)
10. **Suzanne 3D Render** - 8 images (1920×1080)
11. **Piano Render** - 5 images (4000×2250)

Each dataset requires images to be placed in the corresponding `data/<dataset_id>/` directory.

## Project Structure

```
.
├── main.py                 # Main pipeline orchestration
├── camera.py               # Camera-related functions
├── epipolar.py            # Epipolar geometry and essential matrix estimation
├── sift.py                # SIFT feature detection and matching
├── misc.py                # Utility functions (triangulation, MSAC, etc.)
├── get_dataset_info.py    # Dataset configurations
└── data/                  # Image datasets
    ├── 1/
    ├── 2/
    └── ...
```

## Algorithm Pipeline

The reconstruction follows these steps:

1. **Feature Detection**: Extract SIFT keypoints and descriptors from all images
2. **Rotation Estimation**: Compute relative rotations between consecutive image pairs
3. **Initial 3D Points**: Triangulate points from an initial image pair
4. **Camera Pose Estimation**: Estimate translation vectors for all cameras using the known rotations
5. **3D Point Triangulation**: Triangulate points between consecutive image pairs
6. **Visualization**: Display the reconstructed 3D point cloud and camera positions

### Key Algorithms

#### Essential Matrix Estimation
- **5-point algorithm**: Minimal solver for calibrated cameras
- **8-point algorithm**: Linear solution with SVD
- **Homography-based**: Extracts essential matrix from homography decomposition
- **MSAC**: Robust estimation to handle outliers

#### Camera Extraction
- Decomposes essential matrix into rotation and translation
- Uses cheirality condition to select the correct camera configuration

#### 3D Triangulation
- Direct Linear Transform (DLT) for multiple point correspondences
- Filters outliers based on reprojection error and distance thresholds

#### Non-linear Refinement
- Levenberg-Marquardt optimization
- Minimizes reprojection error with respect to camera parameters

## Implementation Details

### Robust Estimation with MSAC

The MSAC (M-estimator Sample Consensus) algorithm is used throughout for robust parameter estimation:
- Handles outliers in feature matches
- Uses T(d,d) early bailout test
- Adaptively updates iteration count based on inlier ratio

### Camera Coordinate System

- First camera is at the identity position: P1 = [I | 0]
- Subsequent cameras are computed relative to the first
- Points are triangulated in the coordinate frame of the first camera

### Normalization

- Image points are calibrated (normalized by intrinsic matrix K)
- Essential matrix enforces singular values [1, 1, 0]
- Homogeneous coordinates are normalized using `pflat()`

## Output

The program generates:
- Console output showing progress and statistics
- 3D visualization with:
  - Colored point cloud (different colors for each image pair)
  - Red camera frustums showing position and orientation
  - Interactive 3D plot (can be rotated with mouse)

## Configuration

Each dataset in `get_dataset_info.py` specifies:
- Image file paths
- Image dimensions
- Focal length (35mm equivalent)
- Initial image pair for bootstrapping
- Pixel threshold for inlier detection

## Notes

- The code assumes calibrated cameras (intrinsic parameters known from EXIF)
- Sequential image processing works best with sufficient overlap between views
- The initial image pair selection is critical for reconstruction quality
- Reprojection error thresholds can be adjusted per dataset
