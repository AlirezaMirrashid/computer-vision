"""
Video Stabilization using Motion Estimation

This module implements various video stabilization techniques based on motion estimation
algorithms described in computer vision literature. It provides a flexible framework
to choose between feature-based, optical flow-based, and parametric motion-based
stabilization methods.

License: MIT
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import argparse
import sys
import json


class VideoStabilizer:
    """
    A comprehensive video stabilization system with multiple motion estimation algorithms.
    
    This class allows selection between different stabilization approaches:
    - Feature-based (e.g., ORB, SIFT) for robust global motion.
    - Optical Flow-based (e.g., Lucas-Kanade) for dense motion fields.
    - Parametric Motion-based (e.g., Affine, Euclidean) for specific global transformations.
    """
    
    def __init__(self, 
                 algorithm: str = 'feature',
                 feature_detector: str = 'ORB',
                 max_features: int = 1000,
                 smoothing_radius: int = 30,
                 border_mode: str = 'black',
                 crop_ratio: float = 0.1,
                 optical_flow_params: Optional[Dict[str, Any]] = None,
                 parametric_motion_type: str = 'euclidean'): # Changed default to euclidean for consistency
        """
        Initialize the video stabilizer with a chosen algorithm.
        
        Args:
            algorithm: The stabilization algorithm to use ('feature', 'optical_flow', 'parametric').
            feature_detector: Type of feature detector for 'feature' algorithm ('ORB', 'SIFT', 'SURF').
            max_features: Maximum number of features for 'feature' or 'optical_flow' algorithm.
            smoothing_radius: Radius for trajectory smoothing.
            border_mode: How to handle borders ('black', 'replicate', 'reflect').
            crop_ratio: Ratio of frame to crop for stability.
            optical_flow_params: Dictionary of parameters for optical flow (e.g., 'winSize', 'maxLevel').
            parametric_motion_type: Type of parametric motion for 'parametric' algorithm ('homography', 'affine', 'euclidean').
        """
        self.algorithm = algorithm
        self.smoothing_radius = smoothing_radius
        self.border_mode = border_mode
        self.crop_ratio = crop_ratio
        
        self.feature_detector_type = feature_detector
        self.max_features = max_features
        self.optical_flow_params = optical_flow_params if optical_flow_params is not None else {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        self.parametric_motion_type = parametric_motion_type
        
        # Initialize algorithm-specific components
        self._init_algorithm_components()
        
        # Motion tracking variables
        self.transforms = [] # Stores relative transforms (dx, dy, da, ds)
        self.absolute_transforms = [] # Stores accumulated 3x3 absolute transforms
        self.trajectory = [] # Stores decomposed absolute trajectory (dx, dy, da, ds)
        self.smoothed_trajectory = [] # Stores smoothed decomposed trajectory
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _init_algorithm_components(self):
        """Initialize components based on the selected algorithm."""
        if self.algorithm == 'feature':
            if self.feature_detector_type == 'ORB':
                self.detector = cv2.ORB_create(nfeatures=self.max_features)
            elif self.feature_detector_type == 'SIFT':
                self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            elif self.feature_detector_type == 'SURF':
                # SURF is patented, ensure it's available or use alternative
                self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            else:
                raise ValueError(f"Unsupported feature detector: {self.feature_detector_type}")
                
            if self.feature_detector_type == 'ORB':
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif self.algorithm == 'optical_flow':
            # No specific initialization needed for optical flow beyond parameters
            pass
        elif self.algorithm == 'parametric':
            # No specific initialization needed for parametric motion beyond type
            pass
        else:
            raise ValueError(f"Unsupported stabilization algorithm: {self.algorithm}")

    def _estimate_relative_transform(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate the relative 3x3 transformation matrix between two frames
        based on the selected algorithm.
        """
        if self.algorithm == 'feature':
            kp1, desc1 = self.detector.detectAndCompute(img1, None)
            kp2, desc2 = self.detector.detectAndCompute(img2, None)
            
            if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
                self.logger.warning("No features detected in one or both frames for feature-based motion.")
                return None
            
            matches = self.matcher.match(desc1, desc2)
            if len(matches) < 10:
                self.logger.warning(f"Insufficient matches found ({len(matches)}) for feature-based motion.")
                return None
            
            matches = sorted(matches, key=lambda x: x.distance)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            if len(pts1) < 4 or len(pts2) < 4:
                return None
                
            # Use estimateAffinePartial2D for Euclidean transform (translation, rotation, uniform scale)
            # This is consistent with the decompose_transform logic
            M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if M is not None:
                return np.vstack([M, [0, 0, 1]]) # Convert 2x3 to 3x3
            return None

        elif self.algorithm == 'optical_flow':
            p0 = cv2.goodFeaturesToTrack(img1, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10)
            if p0 is None or len(p0) == 0:
                self.logger.warning("No good features to track for optical flow.")
                return None

            p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **self.optical_flow_params)

            if p1 is None or st is None or len(p1) == 0:
                self.logger.warning("Optical flow calculation failed or returned no points.")
                return None

            good_new = p1[st==1]
            good_old = p0[st==1]

            if len(good_new) < 4 or len(good_old) < 4:
                self.logger.warning(f"Insufficient good points ({len(good_new)}) for optical flow motion estimation.")
                return None

            M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if M is not None:
                return np.vstack([M, [0, 0, 1]]) # Convert 2x3 to 3x3
            return None

        elif self.algorithm == 'parametric':
            # For parametric, we still need points to estimate the model
            # Using ORB features for correspondence, then estimating the chosen parametric model
            detector = cv2.ORB_create(nfeatures=self.max_features)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            kp1, desc1 = detector.detectAndCompute(img1, None)
            kp2, desc2 = detector.detectAndCompute(img2, None)

            if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
                self.logger.warning("No features detected for parametric motion estimation.")
                return None

            matches = matcher.match(desc1, desc2)
            if len(matches) < 10:
                self.logger.warning(f"Insufficient matches found ({len(matches)}) for parametric motion estimation.")
                return None

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            if self.parametric_motion_type == 'homography':
                if len(pts1) < 4 or len(pts2) < 4:
                    return None
                transform, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=5.0)
            elif self.parametric_motion_type == 'affine':
                if len(pts1) < 3 or len(pts2) < 3:
                    self.logger.warning("Insufficient points for affine transformation (need at least 3).")
                    return None
                transform, _ = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if transform is not None:
                    transform = np.vstack([transform, [0, 0, 1]])
            elif self.parametric_motion_type == 'euclidean':
                if len(pts1) < 2 or len(pts2) < 2:
                    self.logger.warning("Insufficient points for euclidean transformation (need at least 2).")
                    return None
                transform, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if transform is not None:
                    transform = np.vstack([transform, [0, 0, 1]])
            else:
                raise ValueError(f"Unsupported parametric motion type: {self.parametric_motion_type}")
                
            return transform
        return None
    
    def decompose_transform(self, transform: np.ndarray) -> Dict[str, float]:
        """
        Decompose a 3x3 transformation matrix into translation, rotation, and scale.
        This assumes a Euclidean/Similarity transform (translation, rotation, uniform scale).
        
        Args:
            transform: 3x3 transformation matrix (e.g., from estimateAffinePartial2D or homography if it's Euclidean-like)
            
        Returns:
            Dictionary with dx, dy, da (angle), ds (scale)
        """
        if transform is None:
            return {'dx': 0, 'dy': 0, 'da': 0, 'ds': 1}
        
        # For a Euclidean/Similarity matrix M = [[s*cos(a), -s*sin(a), tx], [s*sin(a), s*cos(a), ty], [0, 0, 1]]
        dx = transform[0, 2]
        dy = transform[1, 2]
        
        a = transform[0, 0]
        b = transform[0, 1]
        
        ds = np.sqrt(a*a + b*b) # Scale
        da = np.arctan2(b, a)   # Rotation angle
        
        return {'dx': dx, 'dy': dy, 'da': da, 'ds': ds}
    
    def smooth_trajectory(self, trajectory: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Smooth the camera trajectory (sequence of decomposed motion parameters)
        using a moving average filter.
        
        Args:
            trajectory: List of motion parameters (dx, dy, da, ds) for each frame.
            
        Returns:
            Smoothed trajectory.
        """
        if len(trajectory) == 0:
            return []
        
        smoothed = []
        
        for i in range(len(trajectory)):
            # Define window for smoothing
            start_idx = max(0, i - self.smoothing_radius)
            end_idx = min(len(trajectory), i + self.smoothing_radius + 1)
            
            # Calculate moving average for each parameter
            window = trajectory[start_idx:end_idx]
            
            avg_dx = np.mean([t['dx'] for t in window])
            avg_dy = np.mean([t['dy'] for t in window])
            avg_da = np.mean([t['da'] for t in window])
            avg_ds = np.mean([t['ds'] for t in window])
            
            smoothed.append({
                'dx': avg_dx,
                'dy': avg_dy, 
                'da': avg_da,
                'ds': avg_ds
            })
        
        return smoothed
    
    def _create_transform_matrix(self, params: Dict[str, float]) -> np.ndarray:
        """
        Create a 3x3 Euclidean transformation matrix from decomposed parameters.
        """
        cos_a = np.cos(params['da'])
        sin_a = np.sin(params['da'])
        scale = params['ds']
        
        T = np.array([
            [scale * cos_a, -scale * sin_a, params['dx']],
            [scale * sin_a, scale * cos_a, params['dy']],
            [0, 0, 1]
        ], dtype=np.float32)
        return T

    def apply_stabilization(self, 
                           frame: np.ndarray, 
                           stabilization_matrix: np.ndarray) -> np.ndarray:
        """
        Apply stabilization transform to a frame.
        
        Args:
            frame: Input frame
            stabilization_matrix: 3x3 transformation matrix to apply
            
        Returns:
            Stabilized frame
        """
        h, w = frame.shape[:2]
        
        # Apply transformation
        if self.border_mode == 'black':
            border_mode = cv2.BORDER_CONSTANT
            border_value = 0
        elif self.border_mode == 'replicate':
            border_mode = cv2.BORDER_REPLICATE
            border_value = None
        else:
            border_mode = cv2.BORDER_REFLECT
            border_value = None
        
        if border_value is not None:
            stabilized = cv2.warpAffine(frame, stabilization_matrix[:2], (w, h), 
                                      borderMode=border_mode,
                                      borderValue=border_value)
        else:
            stabilized = cv2.warpAffine(frame, stabilization_matrix[:2], (w, h), 
                                      borderMode=border_mode)
        
        # Apply cropping to remove black borders
        if self.crop_ratio > 0:
            crop_x = int(w * self.crop_ratio)
            crop_y = int(h * self.crop_ratio)
            stabilized = stabilized[crop_y:h-crop_y, crop_x:w-crop_x]
            
            # Resize back to original size
            stabilized = cv2.resize(stabilized, (w, h))
        
        return stabilized
    
    def stabilize_video(self, 
                       input_path: str, 
                       output_path: str,
                       progress_callback: Optional[callable] = None) -> bool:
        """
        Stabilize a video file using the selected algorithm.
        
        Args:
            input_path: Path to input video
            output_path: Path to output stabilized video
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.logger.error(f"Cannot open video: {input_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                self.logger.error("Cannot read first frame")
                return False
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Reset tracking variables for a new stabilization run
            self.transforms = []
            self.absolute_transforms = [np.eye(3)] # Initialize with identity for the first frame
            self.trajectory = []
            self.smoothed_trajectory = []
            
            frame_count = 0
            
            # First pass: estimate motion and accumulate absolute transforms
            self.logger.info(f"First pass: Estimating motion using {self.algorithm} algorithm...")
            
            # Loop through frames to estimate relative transforms and accumulate absolute transforms
            # We process total_frames - 1 pairs of frames (prev_frame, curr_frame)
            for _ in range(int(total_frames) - 1):
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Estimate relative transform (T_rel)
                relative_transform_matrix = self._estimate_relative_transform(prev_gray, curr_gray)
                
                if relative_transform_matrix is None:
                    # If estimation fails, assume identity transform (no motion)
                    relative_transform_matrix = np.eye(3)
                    self.logger.warning(f"Motion estimation failed for frame {frame_count}. Assuming identity transform.")

                # Accumulate absolute transform: T_abs_curr = T_abs_prev @ T_rel
                current_absolute_transform = self.absolute_transforms[-1] @ relative_transform_matrix
                self.absolute_transforms.append(current_absolute_transform)
                
                # Decompose and store for trajectory smoothing
                decomposed_params = self.decompose_transform(current_absolute_transform)
                self.trajectory.append(decomposed_params)
                
                prev_gray = curr_gray.copy()
                frame_count += 1
                
                if progress_callback:
                    progress_callback(frame_count, total_frames - 1, "Analyzing motion")
            
            # Smooth trajectory
            self.logger.info("Smoothing trajectory...")
            # The trajectory list has total_frames - 1 elements (for frames 1 to total_frames - 1)
            # The smoothed_trajectory will also have total_frames - 1 elements
            self.smoothed_trajectory = self.smooth_trajectory(self.trajectory)
            
            # Second pass: apply stabilization
            self.logger.info("Second pass: Applying stabilization...")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            ret, frame = cap.read()
            out.write(frame)  # Write first frame as-is (no stabilization needed for first frame)
            
            frame_count = 0
            
            # Loop through frames to apply stabilization
            # We need to process total_frames - 1 frames after the first one.
            # The indices for absolute_transforms and smoothed_trajectory will go from 0 to total_frames - 2
            # corresponding to the frames from the second frame to the last frame.
            for i in range(int(total_frames) - 1): 
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get original shaky absolute transform for current frame
                # This corresponds to the (i+1)-th accumulated transform in absolute_transforms
                T_shaky = self.absolute_transforms[i+1]
                
                # Construct the new smooth absolute transform for current frame
                # This corresponds to the i-th smoothed trajectory element
                T_smooth = self._create_transform_matrix(self.smoothed_trajectory[i])
                
                # Calculate the final correction matrix: T_correction = T_smooth @ inverse(T_shaky)
                # Use np.linalg.inv for matrix inverse
                T_correction = T_smooth @ np.linalg.inv(T_shaky)
                
                # Apply stabilization
                stabilized_frame = self.apply_stabilization(frame, T_correction)
                out.write(stabilized_frame)
                
                frame_count += 1
                
                if progress_callback:
                    progress_callback(frame_count, total_frames - 1, "Stabilizing")
            
            # Cleanup
            cap.release()
            out.release()
            
            self.logger.info(f"Video stabilization completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during video stabilization: {str(e)}")
            return False
    
    def get_stabilization_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics to evaluate stabilization quality.
        
        Returns:
            Dictionary with stabilization metrics
        """
        if not self.trajectory or not self.smoothed_trajectory:
            return {}
        
        # Convert lists of dicts to numpy arrays for easier calculation
        orig_dx = np.array([t['dx'] for t in self.trajectory])
        orig_dy = np.array([t['dy'] for t in self.trajectory])
        orig_da = np.array([t['da'] for t in self.trajectory])
        orig_ds = np.array([t['ds'] for t in self.trajectory])

        smooth_dx = np.array([t['dx'] for t in self.smoothed_trajectory])
        smooth_dy = np.array([t['dy'] for t in self.smoothed_trajectory])
        smooth_da = np.array([t['da'] for t in self.smoothed_trajectory])
        smooth_ds = np.array([t['ds'] for t in self.smoothed_trajectory])

        # Calculate variance for each component
        orig_variance_x = np.var(orig_dx)
        orig_variance_y = np.var(orig_dy)
        orig_variance_angle = np.var(orig_da)
        orig_variance_scale = np.var(orig_ds)

        smooth_variance_x = np.var(smooth_dx)
        smooth_variance_y = np.var(smooth_dy)
        smooth_variance_angle = np.var(smooth_da)
        smooth_variance_scale = np.var(smooth_ds)

        # Calculate motion reduction (1 - (smoothed_variance / original_variance))
        # Add a small epsilon to avoid division by zero if original variance is 0
        motion_reduction_x = 1 - (smooth_variance_x / (orig_variance_x + 1e-8))
        motion_reduction_y = 1 - (smooth_variance_y / (orig_variance_y + 1e-8))
        motion_reduction_angle = 1 - (smooth_variance_angle / (orig_variance_angle + 1e-8))
        motion_reduction_scale = 1 - (smooth_variance_scale / (orig_variance_scale + 1e-8))
        
        return {
            'motion_reduction_x': float(motion_reduction_x),
            'motion_reduction_y': float(motion_reduction_y),
            'motion_reduction_angle': float(motion_reduction_angle),
            'motion_reduction_scale': float(motion_reduction_scale),
            'original_variance_x': float(orig_variance_x),
            'original_variance_y': float(orig_variance_y),
            'original_variance_angle': float(orig_variance_angle),
            'original_variance_scale': float(orig_variance_scale),
            'smoothed_variance_x': float(smooth_variance_x),
            'smoothed_variance_y': float(smooth_variance_y),
            'smoothed_variance_angle': float(smooth_variance_angle),
            'smoothed_variance_scale': float(smooth_variance_scale),
            'total_frames': len(self.trajectory)
        }


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def validate_input_file(file_path: str) -> bool:
    """Validate that input file exists and is a video file."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: Input file '{file_path}' does not exist.")
        return False
    
    if not path.is_file():
        print(f"Error: '{file_path}' is not a file.")
        return False
    
    # Check file extension
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    if path.suffix.lower() not in video_extensions:
        print(f"Warning: '{file_path}' may not be a supported video format.")
        print(f"Supported formats: {', '.join(video_extensions)}")
    
    return True


def validate_output_path(file_path: str) -> bool:
    """Validate output path and create directories if needed."""
    path = Path(file_path)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if path.exists():
        response = input(f"Output file '{file_path}' already exists. Overwrite? (y/N): ")
        if response.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return False
    
    return True


def progress_callback(current: int, total: int, stage: str):
    """Progress callback for video processing."""
    progress = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r{stage}: |{bar}| {progress:.1f}% ({current}/{total})', end='', flush=True)
    
    if current == total:
        print()  # New line when complete


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Stabilize shaky videos using motion estimation techniques.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %python video_stabilizer.py input.mp4 output.mp4
  %python video_stabilizer.py  input.mp4 output.mp4 --algorithm optical_flow
  %(python video_stabilizer.py  input.mp4 output.mp4 --algorithm parametric --parametric-motion-type affine
  %python video_stabilizer.py  input.mp4 output.mp4 --detector SIFT --smoothing 50
  %python video_stabilizer.py  input.mp4 output.mp4 --crop 0.1 --border replicate
  %python video_stabilizer.py  input.mp4 output.mp4 --config config.json
        """
    )
    
    # Required arguments
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', help='Output video file path')
    
    # Optional arguments
    parser.add_argument(
        '--algorithm',
        choices=['feature', 'optical_flow', 'parametric'],
        default='feature',
        help='Stabilization algorithm to use (default: feature)'
    )
    
    parser.add_argument(
        '--detector',
        choices=['ORB', 'SIFT', 'SURF'],
        default='ORB',
        help='Feature detector to use for \'feature\' algorithm (default: ORB)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=1000,
        help='Maximum number of features to detect for \'feature\' or \'optical_flow\' algorithms (default: 1000)'
    )
    
    parser.add_argument(
        '--smoothing',
        type=int,
        default=30,
        help='Smoothing radius for trajectory stabilization (default: 30)'
    )
    
    parser.add_argument(
        '--crop',
        type=float,
        default=0.05,
        help='Crop ratio to remove black borders (default: 0.05)'
    )
    
    parser.add_argument(
        '--border',
        choices=['black', 'replicate', 'reflect'],
        default='black',
        help='Border handling mode (default: black)'
    )
    
    parser.add_argument(
        '--parametric-motion-type',
        choices=['homography', 'affine', 'euclidean'],
        default='homography',
        help='Parametric motion type for \'parametric\' algorithm (default: homography)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='JSON configuration file path'
    )
    
    parser.add_argument(
        '--metrics',
        action='store_true',
        help='Save stabilization metrics to a JSON file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Video Stabilizer 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration from file if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            return 1
    
    # Override config with command line arguments
    stabilizer_config = {
        'algorithm': config.get('algorithm', args.algorithm),
        'feature_detector': config.get('feature_detector', args.detector),
        'max_features': config.get('max_features', args.max_features),
        'smoothing_radius': config.get('smoothing_radius', args.smoothing),
        'border_mode': config.get('border_mode', args.border),
        'crop_ratio': config.get('crop_ratio', args.crop),
        'parametric_motion_type': config.get('parametric_motion_type', args.parametric_motion_type)
    }
    
    # Validate inputs
    if not validate_input_file(args.input):
        return 1
    
    if not validate_output_path(args.output):
        return 1
    
    # Create stabilizer
    try:
        stabilizer = VideoStabilizer(**stabilizer_config)
        logger.info("Video stabilizer initialized successfully")
        logger.info(f"Configuration: {stabilizer_config}")
    except Exception as e:
        logger.error(f"Failed to initialize video stabilizer: {e}")
        return 1
    
    # Perform stabilization
    print(f"Stabilizing video: {args.input} -> {args.output}")
    print("This may take a while depending on video length and resolution...")
    
    try:
        success = stabilizer.stabilize_video(
            args.input,
            args.output,
            progress_callback=progress_callback
        )
        
        if success:
            print(f"\n✓ Video stabilization completed successfully!")
            print(f"Output saved to: {args.output}")
            
            # Save metrics if requested
            if args.metrics:
                metrics = stabilizer.get_stabilization_metrics()
                metrics_file = Path(args.output).with_suffix('.metrics.json')
                
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"Metrics saved to: {metrics_file}")
                print(f"Motion reduction: X={metrics.get('motion_reduction_x', 0):.2%}, "
                      f"Y={metrics.get('motion_reduction_y', 0):.2%}")
            
            return 0
            
        else:
            print(f"\n✗ Video stabilization failed!")
            logger.error("Stabilization process returned failure status")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Error during stabilization: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())



