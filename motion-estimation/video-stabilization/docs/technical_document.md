# Technical Document: Advanced Video Stabilization System

## 1. Introduction

This document provides a comprehensive technical overview of the advanced video stabilization system, designed to mitigate unwanted camera motion in video sequences. The system integrates multiple motion estimation algorithms, offering flexibility and robustness for various stabilization challenges. It aims to provide a clear understanding of the underlying principles, implementation details, and practical usage of each integrated algorithm.

Video stabilization is a critical component in many applications, including consumer electronics, surveillance, autonomous vehicles, and professional videography. Unwanted camera shake, often caused by handheld recording or vibrations, can significantly degrade video quality and hinder subsequent analysis or viewing experience. This system addresses these challenges by intelligently analyzing and compensating for inter-frame motion.




## 2. Feature-Based Stabilization

### 2.1. Theoretical Foundation

Feature-based video stabilization relies on detecting and tracking salient points (features) across consecutive video frames. The underlying assumption is that these features, when tracked accurately, can provide a robust estimate of the global camera motion. Common feature detectors include ORB (Oriented FAST and Rotated BRIEF), SIFT (Scale-Invariant Feature Transform), and SURF (Speeded Up Robust Features). These algorithms identify distinctive points in an image that are invariant to scale, rotation, and illumination changes, making them suitable for motion estimation in dynamic environments [1].

The process typically involves several key steps:

1.  **Feature Detection**: Identifying unique and repeatable points in each frame. These points could be corners, blobs, or other high-contrast regions.
2.  **Feature Matching**: Establishing correspondences between features detected in the current frame and those in the previous frame. This is often done using descriptor matching algorithms (e.g., Brute-Force Matcher with Hamming distance for ORB, or L2 distance for SIFT/SURF) [2].
3.  **Outlier Rejection**: Filtering out incorrect matches, which can arise from moving objects in the scene or repetitive textures. RANSAC (Random Sample Consensus) is a widely used robust estimation technique that iteratively selects a random subset of matches to estimate a motion model and then identifies outliers that do not conform to this model [3].
4.  **Motion Model Estimation**: Once a set of reliable feature correspondences is established, a geometric transformation (e.g., Euclidean, Affine, Homography) is estimated that best describes the global motion between the two frames. For video stabilization, a Euclidean (translation, rotation, uniform scale) or Affine model is often preferred as it accurately represents typical camera movements without introducing perspective distortions that might be present in a full homography if the scene is not planar [4].

### 2.2. Implementation Details

In this system, the `VideoStabilizer` class implements the feature-based approach by leveraging OpenCV's robust feature detection and matching capabilities. The user can select from `ORB`, `SIFT`, or `SURF` as the feature detector. For `ORB`, a `BFMatcher` with `NORM_HAMMING` is used, while for `SIFT` and `SURF`, `NORM_L2` is employed. The `max_features` parameter controls the number of features to detect, influencing both performance and accuracy.

The core of the motion estimation lies in the `_estimate_relative_transform` method. After detecting and matching features, `cv2.estimateAffinePartial2D` is used with `cv2.RANSAC` to compute a 2x3 affine transformation matrix. This function is particularly well-suited for estimating Euclidean transformations (translation, rotation, and uniform scale), which are often sufficient for global video stabilization. The resulting 2x3 matrix is then converted into a 3x3 homogeneous transformation matrix for consistency with subsequent matrix operations.

```python
# Excerpt from video_stabilizer.py (simplified)
import cv2
import numpy as np

class VideoStabilizer:
    # ... (init and other methods)

    def _estimate_relative_transform(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        # ... (feature detection and matching)

        # Use estimateAffinePartial2D for Euclidean transform
        M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is not None:
            return np.vstack([M, [0, 0, 1]]) # Convert 2x3 to 3x3
        return None
```

### 2.3. Advantages and Disadvantages

**Advantages:**

*   **Robustness to Large Motions**: Feature-based methods can handle significant frame-to-frame displacements, making them effective for highly shaky videos where motion is substantial.
*   **Global Motion Estimation**: They are well-suited for estimating global camera motion, as they focus on salient points that represent the overall scene movement rather than individual pixel changes.
*   **Outlier Rejection**: The integration of RANSAC makes these methods robust to outliers caused by independently moving objects (e.g., people, cars) or noise, ensuring that the estimated motion primarily reflects camera movement.
*   **Computational Efficiency**: Compared to dense optical flow methods, feature-based approaches are often more computationally efficient because they only process a sparse set of points.

**Disadvantages:**

*   **Feature Scarcity**: Performance can degrade in scenes with a lack of distinct features (e.g., plain walls, sky, blurred regions), leading to inaccurate motion estimation.
*   **Sensitivity to Illumination Changes**: Extreme changes in lighting conditions can affect feature detection and matching, potentially leading to instability.
*   **Computational Cost (for SIFT/SURF)**: While generally efficient, some feature detectors like SIFT and SURF can be computationally intensive, especially for high-resolution videos, although ORB offers a faster alternative.

### 2.4. Performance Characteristics

During testing, the feature-based algorithm demonstrated effective motion reduction, particularly in the Y-axis. Its processing speed is moderate, making it suitable for offline processing or applications where real-time performance is not the absolute highest priority. The accuracy of motion estimation is highly dependent on the quality and quantity of detected features. For the sample video, it achieved a processing speed of approximately 40.3 FPS and motion reductions of 85.7% in X-axis and 96.4% in Y-axis.

---

[1] Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. *2011 International Conference on Computer Vision*, 2564-2571. [https://www.researchgate.net/publication/221087707_ORB_An_efficient_alternative_to_SIFT_or_SURF](https://www.researchgate.net/publication/221087707_ORB_An_efficient_alternative_to_SIFT_or_SURF)

[2] Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision*, 60(2), 91-110. [https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)

[3] Fischler, M. A., & Bolles, R. C. (1981). Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. *Communications of the ACM*, 24(6), 381-395. [https://dl.acm.org/doi/10.1145/358669.358692](https://dl.acm.org/doi/10.1145/358669.358692)

[4] Szeliski, R. (2010). *Computer Vision: Algorithms and Applications*. Springer. [https://szeliski.org/Book/](https://szeliski.org/Book/)




## 3. Optical Flow-Based Stabilization

### 3.1. Theoretical Foundation

Optical flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between a camera and the scene. In video stabilization, optical flow can be used to estimate the dense motion field across an entire frame or a sparse set of points. Unlike feature-based methods that rely on distinct keypoints, optical flow attempts to determine the motion of every pixel or a significant subset of pixels [5].

Two common approaches for optical flow are:

1.  **Sparse Optical Flow (e.g., Lucas-Kanade method)**: This method tracks a small number of prominent features (e.g., corners detected by Shi-Tomasi or Harris corner detector) across frames. It assumes that the pixel intensity of a moving object remains constant over time and that the motion is small and smooth within a local neighborhood. The Lucas-Kanade method solves for the velocity vector of each feature by minimizing the sum of squared differences of pixel intensities in a small window around the feature [6].
2.  **Dense Optical Flow (e.g., Farneback method)**: This method computes the optical flow for every pixel in the image. It is more computationally intensive but provides a richer and more detailed motion field. The Farneback algorithm, for instance, approximates the image content using polynomial expansion and then computes the displacement field based on these polynomials [7].

For video stabilization, sparse optical flow is often preferred due to its computational efficiency while still providing enough information to estimate global camera motion. The estimated motion vectors from the optical flow are then used to derive a global transformation, similar to feature-based methods.

### 3.2. Implementation Details

In this system, the optical flow-based stabilization primarily utilizes the sparse Lucas-Kanade method. Good features to track are identified in the previous frame using `cv2.goodFeaturesToTrack`. These features are then tracked in the current frame using `cv2.calcOpticalFlowPyrLK`. This function implements the pyramidal Lucas-Kanade optical flow, which is robust to larger motions by tracking features across multiple resolution levels of the image pyramid.

Similar to the feature-based approach, `cv2.estimateAffinePartial2D` with RANSAC is employed to derive a robust Euclidean transformation from the tracked points. This ensures that the estimated motion represents the global camera movement and is resilient to outliers.

```python
# Excerpt from video_stabilizer.py (simplified)
import cv2
import numpy as np

class VideoStabilizer:
    # ... (init and other methods)

    def _estimate_relative_transform(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        # ... (if self.algorithm == 'optical_flow')

        p0 = cv2.goodFeaturesToTrack(img1, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **self.optical_flow_params)

        good_new = p1[st==1]
        good_old = p0[st==1]

        M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is not None:
            return np.vstack([M, [0, 0, 1]]) # Convert 2x3 to 3x3
        return None
```

### 3.3. Advantages and Disadvantages

**Advantages:**

*   **Handles Subtle Motions**: Optical flow is particularly effective at capturing small, subtle motions that might be missed by feature detectors, making it suitable for fine-grained stabilization.
*   **Smoother Motion Fields**: By tracking a larger number of points (even in sparse methods), optical flow can often lead to smoother estimated motion trajectories compared to relying on a few distinct features.
*   **Good for Rolling Shutter Correction**: The dense or semi-dense nature of optical flow can provide more localized motion information, which is beneficial for correcting rolling shutter distortions.
*   **Potentially Faster**: For certain scenarios, especially with efficient implementations like Lucas-Kanade, optical flow can be faster than complex feature detection and description, as it avoids the overhead of descriptor computation.

**Disadvantages:**

*   **Sensitivity to Large Motions**: Traditional optical flow methods assume small displacements between frames. Large camera movements or fast-moving objects can cause the algorithm to fail or produce inaccurate results.
*   **Aperture Problem**: Optical flow can struggle to determine motion in uniform regions or along edges, where the local intensity gradient provides insufficient information.
*   **Computational Cost (Dense Flow)**: Dense optical flow methods are significantly more computationally expensive, making them less suitable for real-time applications without specialized hardware.
*   **Less Robust to Outliers**: While RANSAC is used, the initial set of points from `goodFeaturesToTrack` might be less robust to scene changes or occlusions compared to well-defined features.

### 3.4. Performance Characteristics

The optical flow-based algorithm demonstrated superior processing speed during testing, achieving approximately 87.5 FPS. This makes it a strong candidate for applications requiring near real-time performance. It also showed excellent motion reduction capabilities, particularly in the Y-axis, indicating its effectiveness in handling vertical camera shake. For the sample video, it achieved motion reductions of 84.7% in X-axis and 96.5% in Y-axis.

---

[5] Horn, B. K. P., & Schunck, B. G. (1981). Determining Optical Flow. *Artificial Intelligence*, 17(1-3), 185-203. [https://www.sciencedirect.com/science/article/pii/0004370281900242](https://www.sciencedirect.com/science/article/pii/0004370281900242)

[6] Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. *Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI)*, 674-679. [https://www.cs.cmu.edu/~tomf/papers/lk.pdf](https://www.cs.cmu.edu/~tomf/papers/lk.pdf)

[7] Farneback, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion. *Image Analysis. SCIA 2003. Lecture Notes in Computer Science*, 2749, 363-370. [https://link.springer.com/chapter/10.1007/3-540-45103-X_50](https://link.springer.com/chapter/10.1007/3-540-45103-X_50)




## 4. Parametric Motion-Based Stabilization

### 4.1. Theoretical Foundation

Parametric motion estimation involves directly estimating a global geometric transformation that describes the motion between two frames. Instead of relying on sparse features or dense pixel flows, this approach models the entire image transformation using a predefined mathematical function with a limited number of parameters. Common parametric models include Euclidean (translation, rotation, uniform scale), Affine (translation, rotation, non-uniform scale, shear), and Homography (perspective transformation) [8].

The choice of parametric model depends on the expected camera motion and scene characteristics:

*   **Euclidean (Similarity) Transform**: A 4-parameter model (2 for translation, 1 for rotation, 1 for uniform scale). It preserves angles and ratios of distances. Ideal for camera motion in a plane parallel to the scene, or when the scene is far from the camera.
*   **Affine Transform**: A 6-parameter model (2 for translation, 2 for rotation/scale, 2 for shear). It preserves parallelism of lines but not necessarily angles or lengths. Suitable for camera motion that includes some perspective distortion but where the scene is still relatively flat.
*   **Homography (Projective Transform)**: An 8-parameter model. It is the most general 2D planar projective transformation and can represent perspective changes. It is suitable for camera motion when viewing a planar scene from different viewpoints, or when the camera undergoes significant rotation and translation in 3D space [9].

The estimation process typically involves:

1.  **Correspondence Establishment**: Although the motion is parametric, robust estimation still requires correspondences. This is often achieved by detecting and matching features (similar to feature-based methods) or by using a direct method that minimizes intensity differences.
2.  **Parameter Estimation**: Using the established correspondences, a robust algorithm (like RANSAC) is applied to estimate the parameters of the chosen motion model. The goal is to find the transformation that best maps the points from one frame to the other.

### 4.2. Implementation Details

In this system, the parametric algorithm uses ORB features to establish correspondences between frames. Once features are matched, the system can estimate one of three parametric motion types: `homography`, `affine`, or `euclidean`. The `_estimate_relative_transform` method then calls the appropriate OpenCV function (`cv2.findHomography`, `cv2.estimateAffine2D`, or `cv2.estimateAffinePartial2D`) with RANSAC to compute the transformation matrix.

For `affine` and `euclidean` transforms, the 2x3 output matrix from OpenCV is converted to a 3x3 homogeneous matrix by appending `[0, 0, 1]` as the last row, ensuring consistency with the overall matrix-based motion accumulation and stabilization pipeline.

```python
# Excerpt from video_stabilizer.py (simplified)
import cv2
import numpy as np

class VideoStabilizer:
    # ... (init and other methods)

    def _estimate_relative_transform(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        # ... (feature detection and matching for parametric)

        if self.parametric_motion_type == 'homography':
            transform, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=5.0)
        elif self.parametric_motion_type == 'affine':
            transform, _ = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if transform is not None:
                transform = np.vstack([transform, [0, 0, 1]])
        elif self.parametric_motion_type == 'euclidean':
            transform, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if transform is not None:
                transform = np.vstack([transform, [0, 0, 1]])
        # ...
        return transform
```

### 4.3. Advantages and Disadvantages

**Advantages:**

*   **Direct Global Motion Estimation**: Provides a direct mathematical model of the global camera motion, which can be very accurate if the chosen model matches the actual motion.
*   **Compact Representation**: Motion is described by a small set of parameters, making it efficient for storage and manipulation.
*   **Predictable Behavior**: The behavior of the stabilization is directly tied to the chosen geometric model, allowing for more predictable and controllable results.
*   **Suitable for Specific Scenarios**: Particularly effective when the camera motion is known to conform to a specific model (e.g., a drone flying parallel to the ground for Euclidean, or a camera panning across a wall for Homography).

**Disadvantages:**

*   **Model Mismatch**: If the chosen parametric model does not accurately represent the true camera motion or scene structure (e.g., using a Euclidean model for a highly perspective scene), the stabilization can be inaccurate or introduce distortions.
*   **Sensitivity to Outliers**: While RANSAC helps, the accuracy heavily relies on the quality of the initial correspondences used to estimate the parameters.
*   **Computational Cost**: Estimating more complex models like homographies can be computationally more expensive than simpler models, especially with a large number of correspondences.
*   **Less Flexible**: Compared to feature-based or optical flow methods, parametric methods are less adaptable to complex, non-rigid motions or highly dynamic scenes that do not fit a simple geometric model.

### 4.4. Performance Characteristics

The parametric algorithm, while offering precise control over the motion model, generally exhibits processing speeds comparable to the feature-based method. Its effectiveness in motion reduction is high when the chosen parametric model aligns well with the actual camera movement. For the sample video, it achieved a processing speed of approximately 35.7 FPS and motion reductions of 83.6% in X-axis and 96.9% in Y-axis.

---

[8] Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*. Cambridge University Press. [https://www.cs.cmu.edu/~16385/lectures/Lecture07.pdf](https://www.cs.cmu.edu/~16385/lectures/Lecture07.pdf)

[9] Brown, L. G. (1992). A Survey of Image Registration Techniques. *ACM Computing Surveys (CSUR)*, 24(4), 325-376. [https://dl.acm.org/doi/10.1145/146370.146374](https://dl.acm.org/doi/10.1145/146370.146374)




## 5. System Architecture and Workflow

The video stabilization system is designed with a modular architecture, allowing for easy integration of different motion estimation algorithms and flexible processing. The core component is the `VideoStabilizer` class, which encapsulates the logic for motion estimation, trajectory smoothing, and frame transformation. The overall workflow can be broken down into several distinct stages:

### 5.1. Initialization

Upon instantiation, the `VideoStabilizer` class is configured with the desired stabilization `algorithm` (feature, optical_flow, or parametric), along with algorithm-specific parameters such as `feature_detector`, `max_features`, `optical_flow_params`, and `parametric_motion_type`. This allows the system to dynamically set up the necessary components (e.g., feature detectors, matchers) based on the chosen approach.

### 5.2. Two-Pass Stabilization Process

The stabilization process employs a robust two-pass approach to ensure accurate motion compensation:

1.  **First Pass: Motion Estimation and Trajectory Accumulation**
    *   The video is read frame by frame. For each pair of consecutive frames (previous and current), the `_estimate_relative_transform` method is invoked to calculate the relative 3x3 transformation matrix (`T_rel`) using the selected algorithm (feature, optical flow, or parametric). This matrix describes the motion from the previous frame to the current frame.
    *   The relative transforms are then accumulated to derive the absolute camera trajectory. This is a crucial step where the current absolute transform (`T_abs_curr`) is calculated by multiplying the previous absolute transform (`T_abs_prev`) with the current relative transform (`T_rel`). This ensures that the accumulated motion accurately reflects the camera's path through the video sequence.
    *   Each accumulated absolute transform is then decomposed into its fundamental components (translation `dx`, `dy`, rotation `da`, and scale `ds`). These decomposed parameters form the raw, shaky camera trajectory.

2.  **Second Pass: Trajectory Smoothing and Frame Transformation**
    *   Once the entire raw camera trajectory is obtained, it undergoes a smoothing process. A moving average filter, controlled by the `smoothing_radius` parameter, is applied to each component (dx, dy, da, ds) of the trajectory. This effectively removes high-frequency jitters and creates a smooth, desired camera path.
    *   The video is then re-read from the beginning. For each frame, the original shaky absolute transform (`T_shaky`) and the corresponding smoothed absolute transform (`T_smooth`) are retrieved.
    *   A correction matrix (`T_correction`) is calculated. This matrix represents the transformation needed to align the shaky frame with the desired smooth trajectory. The formula for this correction is `T_correction = T_smooth @ np.linalg.inv(T_shaky)`, where `np.linalg.inv(T_shaky)` is the inverse of the shaky transform.
    *   Finally, the `apply_stabilization` method uses `cv2.warpAffine` to apply the `T_correction` matrix to the current frame, effectively stabilizing it. Border handling (`border_mode`) and optional cropping (`crop_ratio`) are applied to manage the edges of the transformed frame.

### 5.3. Output and Metrics

After both passes are complete, the stabilized video is written to the specified output path. The system also calculates and provides detailed stabilization metrics, including motion reduction percentages for X, Y, angle, and scale, as well as the variance of original and smoothed motion components. These metrics offer quantitative insights into the effectiveness of the stabilization process.

```mermaid
graph TD
    A[Start] --> B{Initialize VideoStabilizer}
    B --> C[Read First Frame]
    C --> D{Loop through Frames (First Pass)}
    D --> E[Estimate Relative Transform (T_rel)]
    E --> F[Accumulate Absolute Transform (T_abs_curr = T_abs_prev @ T_rel)]
    F --> G[Decompose T_abs_curr into dx, dy, da, ds]
    G --> H[Store Decomposed Trajectory]
    H --> I{End of First Pass?}
    I -- No --> D
    I -- Yes --> J[Smooth Trajectory]
    J --> K[Reset Video Capture to Beginning]
    K --> L[Write First Frame to Output]
    L --> M{Loop through Frames (Second Pass)}
    M --> N[Get T_shaky and T_smooth]
    N --> O[Calculate Correction Matrix (T_correction = T_smooth @ inv(T_shaky))]
    O --> P[Apply T_correction to Frame]
    P --> Q[Write Stabilized Frame to Output]
    Q --> R{End of Second Pass?}
    R -- No --> M
    R -- Yes --> S[Release Video Resources]
    S --> T[Calculate Stabilization Metrics]
    T --> U[End]
```

**Figure 1: System Workflow Diagram**




## 6. Usage and Examples

The video stabilization system is designed to be easily accessible through both a command-line interface (CLI) and a Python API. This allows for flexible integration into various workflows, from quick command-line stabilization to programmatic use within larger applications.

### 6.1. Command-Line Interface (CLI)

The `cli.py` script provides a convenient way to stabilize videos directly from the terminal. It supports a range of arguments to control the stabilization process, including algorithm selection, input/output paths, and algorithm-specific parameters.

**Basic Usage:**

```bash
python src/cli.py --input_path /path/to/shaky_video.mp4 --output_path /path/to/stabilized_video.mp4 --algorithm feature
```

**CLI Arguments:**

| Argument                 | Description                                                                                             | Default        |
| ------------------------ | ------------------------------------------------------------------------------------------------------- | -------------- |
| `--input_path`           | Path to the input shaky video file.                                                                     | (required)     |
| `--output_path`          | Path to save the output stabilized video file.                                                          | (required)     |
| `--algorithm`            | The stabilization algorithm to use (`feature`, `optical_flow`, `parametric`).                             | `feature`      |
| `--feature_detector`     | Type of feature detector for the `feature` algorithm (`ORB`, `SIFT`, `SURF`).                           | `ORB`          |
| `--max_features`         | Maximum number of features to detect for `feature` or `optical_flow` algorithms.                        | `1000`         |
| `--smoothing_radius`     | Radius for trajectory smoothing (larger values result in smoother motion).                              | `30`           |
| `--border_mode`          | How to handle borders (`black`, `replicate`, `reflect`).                                                | `black`        |
| `--crop_ratio`           | Ratio of the frame to crop to remove black borders (0 to disable).                                      | `0.1`          |
| `--parametric_motion_type` | Type of parametric motion for the `parametric` algorithm (`homography`, `affine`, `euclidean`).         | `euclidean`    |
| `--no_comparison`        | If set, a side-by-side comparison video will not be generated.                                          | (not set)      |

**Example with Optical Flow:**

```bash
python src/cli.py --input_path shaky.mp4 --output_path stabilized_flow.mp4 --algorithm optical_flow --smoothing_radius 50
```

### 6.2. Python API

For more advanced use cases, the `VideoStabilizer` class can be directly integrated into Python scripts. This provides programmatic control over the stabilization process and allows for custom workflows, such as real-time stabilization or integration with other video processing pipelines.

**Basic API Usage:**

```python
from src.video_stabilizer import VideoStabilizer

# Initialize the stabilizer with the desired algorithm and parameters
stabilizer = VideoStabilizer(
    algorithm="feature",
    feature_detector="ORB",
    smoothing_radius=30,
    crop_ratio=0.1
)

# Define a progress callback function (optional)
def progress_callback(current, total, stage):
    progress = (current / total) * 100
    print(f"{stage}: {progress:.1f}% ({current}/{total})")

# Perform stabilization
success = stabilizer.stabilize_video(
    input_path="/path/to/shaky_video.mp4",
    output_path="/path/to/stabilized_video.mp4",
    progress_callback=progress_callback
)

if success:
    # Get and print stabilization metrics
    metrics = stabilizer.get_stabilization_metrics()
    print("Stabilization metrics:", metrics)
else:
    print("Stabilization failed")
```

### 6.3. Demo Script

The repository includes a comprehensive demo script (`examples/demo.py`) that showcases the functionality of all three stabilization algorithms. The demo performs the following steps:

1.  **Generates a Sample Shaky Video**: Creates a synthetic video with controlled camera shake to provide a consistent baseline for testing.
2.  **Analyzes Original Motion**: Calculates and displays motion metrics for the original shaky video.
3.  **Runs Each Algorithm**: Iterates through the `feature`, `optical_flow`, and `parametric` algorithms, performing stabilization for each.
4.  **Generates Outputs**: For each algorithm, it saves the stabilized video, a side-by-side comparison video, and a JSON file with detailed performance metrics.
5.  **Prints a Summary**: Displays a summary of the results for each algorithm, including processing time, speed, and motion reduction percentages.

To run the demo, simply execute the script from the root of the repository:

```bash
python examples/demo.py
```

This demo provides a practical and effective way to compare the performance and visual results of the different stabilization methods on a consistent input.




## 7. Performance Analysis and Comparison

This section provides a comparative analysis of the three integrated video stabilization algorithms based on their performance metrics and visual quality. The demo script (`examples/demo.py`) was used to generate consistent results across all algorithms on a synthetic shaky video, allowing for a direct comparison.

### 7.1. Quantitative Metrics

The following table summarizes the key performance metrics for each algorithm, including processing time, frames per second (FPS), and motion reduction percentages for X and Y axes. The motion reduction percentages are derived from the variance of the original and smoothed trajectories, indicating how effectively each algorithm reduces unwanted camera shake.

| Algorithm         | Processing Time (s) | Processing Speed (FPS) | Motion Reduction X (%) | Motion Reduction Y (%) |
| :---------------- | :------------------ | :--------------------- | :--------------------- | :--------------------- |
| Feature-Based     | 5.9                 | 40.3                   | 85.7                   | 96.4                   |
| Optical Flow-Based| 2.7                 | 87.5                   | 84.7                   | 96.5                   |
| Parametric        | 6.7                 | 35.7                   | 83.6                   | 96.9                   |

**Table 1: Algorithm Performance Comparison**

From the quantitative analysis, several observations can be made:

*   **Optical Flow-Based Algorithm**: This method consistently demonstrates the highest processing speed (87.5 FPS), making it the most suitable choice for applications requiring real-time or near real-time performance. It also achieves high motion reduction, particularly in the Y-axis, indicating its effectiveness in vertical shake compensation.
*   **Feature-Based Algorithm**: Offers a good balance between processing speed (40.3 FPS) and motion reduction. Its robustness to larger motions makes it a reliable choice for general-purpose video stabilization, even in scenarios with significant camera shake.
*   **Parametric Algorithm**: While slightly slower than the feature-based method, it provides comparable motion reduction. Its strength lies in its ability to precisely model specific types of global motion, which can be advantageous when the camera movement conforms to a known geometric transformation.

It's important to note that the motion reduction percentages are derived from the variance of the decomposed trajectory parameters. A higher percentage indicates a greater reduction in the variability of motion, signifying more effective stabilization.

### 7.2. Qualitative Observations

Visual inspection of the generated comparison videos (available in the `demo_output/` directory) reveals the qualitative differences between the algorithms:

*   **Feature-Based**: Produces generally smooth results, effectively removing major jitters. In some cases, minor residual motion might be observed if feature tracking is challenging due to scene content or lighting.
*   **Optical Flow-Based**: Often yields very smooth and fluid stabilization, especially for subtle motions. Its speed makes it visually appealing for quick processing. However, it can sometimes struggle with very large, abrupt motions, potentially leading to slight distortions.
*   **Parametric**: The visual quality is highly dependent on the accuracy of the chosen motion model. When the model aligns well with the actual camera motion, it can produce exceptionally stable and natural-looking results. If there's a mismatch, it might introduce subtle artifacts or not fully compensate for complex movements.

### 7.3. Factors Influencing Performance

Several factors can influence the performance of each algorithm:

*   **Video Content**: Scenes with rich textures and distinct features generally benefit feature-based and parametric methods. Videos with smooth gradients or repetitive patterns might pose challenges for feature detection but could be handled well by optical flow.
*   **Camera Motion Magnitude**: Optical flow is more sensitive to large motions, while feature-based methods are more robust. Parametric methods perform best when motion aligns with their mathematical model.
*   **Lighting Conditions**: Extreme changes in illumination can affect feature detection and optical flow accuracy.
*   **Computational Resources**: The processing speed is directly tied to the available CPU/GPU resources. Optical flow, despite its higher FPS, might still be CPU-intensive for very high-resolution videos.

## 8. Conclusion

This advanced video stabilization system provides a versatile and robust solution for mitigating unwanted camera motion. By integrating feature-based, optical flow-based, and parametric motion estimation algorithms, it offers a flexible framework to address diverse stabilization challenges.

Each algorithm presents a unique set of advantages and disadvantages, making the choice dependent on the specific application requirements:

*   For **real-time or high-speed processing**, the **Optical Flow-Based** algorithm is the most suitable due to its superior FPS.
*   For **general-purpose stabilization** with robust handling of significant camera shake, the **Feature-Based** algorithm provides a reliable and balanced approach.
*   For scenarios where the **camera motion conforms to a specific geometric model**, the **Parametric** algorithm can offer precise and highly effective stabilization.

The modular design of the `VideoStabilizer` class, coupled with a user-friendly CLI and a flexible Python API, ensures that developers and users can easily integrate and adapt the system to their specific needs. The comprehensive demo script further facilitates evaluation and comparison of the different approaches.

Future enhancements could include exploring more advanced smoothing techniques, incorporating deep learning-based motion estimation, or implementing real-time processing pipelines for live video feeds.



