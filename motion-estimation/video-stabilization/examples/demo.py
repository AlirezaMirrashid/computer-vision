"""
Video Stabilization Demo

This script demonstrates the video stabilization functionality by:
1. Creating a synthetic shaky video
2. Applying stabilization using different algorithms (feature, optical flow, parametric)
3. Showing before/after comparison for each
4. Displaying performance metrics for each

License: MIT
"""

import cv2
import numpy as np
import os
import sys
import json
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from video_stabilizer import VideoStabilizer


def create_sample_shaky_video(output_path: str, duration_seconds: int = 10) -> str:
    """
    Create a sample shaky video for demonstration purposes.
    
    Args:
        output_path: Path where to save the sample video
        duration_seconds: Duration of the video in seconds
        
    Returns:
        Path to the created video file
    """
    print("🎬 Creating sample shaky video...")
    
    # Video properties
    fps = 30
    width, height = 640, 480
    total_frames = duration_seconds * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a scene with various objects
    def create_scene():
        scene = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            intensity = int(50 + (y / height) * 100)
            scene[y, :] = [intensity, intensity//2, intensity//3]
        
        # Add geometric shapes
        cv2.rectangle(scene, (100, 100), (200, 180), (0, 255, 0), -1)
        cv2.circle(scene, (350, 150), 50, (255, 0, 0), -1)
        cv2.rectangle(scene, (450, 200), (580, 350), (0, 0, 255), 3)
        cv2.ellipse(scene, (200, 300), (80, 40), 45, 0, 360, (255, 255, 0), -1)
        
        # Add text
        cv2.putText(scene, "STABILIZATION DEMO", (150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add random texture points
        np.random.seed(42)  # For reproducible results
        for _ in range(200):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            color = tuple(map(int, np.random.randint(100, 255, 3)))
            cv2.circle(scene, (x, y), 2, color, -1)
        
        return scene
    
    base_scene = create_scene()
    
    print(f"📹 Generating {total_frames} frames with camera shake...")
    
    for frame_idx in range(total_frames):
        # Simulate realistic camera shake
        t = frame_idx / fps
        
        # Multiple frequency components for realistic shake
        shake_x = (15 * np.sin(2.5 * t) + 
                  8 * np.sin(7.2 * t) + 
                  3 * np.sin(15.8 * t) + 
                  2 * np.random.randn())
        
        shake_y = (12 * np.cos(3.1 * t) + 
                  6 * np.cos(8.7 * t) + 
                  2 * np.cos(18.3 * t) + 
                  1.5 * np.random.randn())
        
        # Add slight rotation shake
        angle = 0.02 * np.sin(4.5 * t) + 0.01 * np.random.randn()
        
        # Create transformation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        rotation_matrix[0, 2] += shake_x
        rotation_matrix[1, 2] += shake_y
        
        # Apply transformation
        shaky_frame = cv2.warpAffine(base_scene, rotation_matrix, (width, height))
        
        # Add some motion blur occasionally
        if frame_idx % 10 == 0:
            kernel = np.ones((3, 3), np.float32) / 9
            shaky_frame = cv2.filter2D(shaky_frame, -1, kernel)
        
        out.write(shaky_frame)
        
        # Progress indicator
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {progress:.1f}%")
    
    out.release()
    print(f"✅ Sample video created: {output_path}")
    return output_path


def analyze_video_motion(video_path: str) -> dict:
    """
    Analyze motion in a video to quantify shakiness.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with motion analysis results
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        return {}
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    motion_vectors = []
    frame_count = 0
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow for motion analysis
        flow = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, 
            np.array([[320, 240]], dtype=np.float32).reshape(-1, 1, 2),
            None
        )[0]
        
        if flow is not None and len(flow) > 0:
            motion_x = flow[0][0][0] - 320
            motion_y = flow[0][0][1] - 240
            motion_vectors.append([motion_x, motion_y])
        
        prev_gray = curr_gray
        frame_count += 1
    
    cap.release()
    
    if not motion_vectors:
        return {}
    
    motion_array = np.array(motion_vectors)
    
    return {
        'mean_motion_x': float(np.mean(np.abs(motion_array[:, 0]))),
        'mean_motion_y': float(np.mean(np.abs(motion_array[:, 1]))),
        'std_motion_x': float(np.std(motion_array[:, 0])),
        'std_motion_y': float(np.std(motion_array[:, 1])),
        'max_motion_x': float(np.max(np.abs(motion_array[:, 0]))),
        'max_motion_y': float(np.max(np.abs(motion_array[:, 1]))),
        'total_frames': frame_count
    }


def create_comparison_video(original_path: str, stabilized_path: str, output_path: str):
    """
    Create a side-by-side comparison video.
    
    Args:
        original_path: Path to original shaky video
        stabilized_path: Path to stabilized video
        output_path: Path for comparison video output
    """
    print("🔄 Creating side-by-side comparison...")
    
    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(stabilized_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Error opening videos for comparison")
        return
    
    # Get video properties
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video (double width for side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    frame_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Add labels
        cv2.putText(frame1, "ORIGINAL (SHAKY)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame2, "STABILIZED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine frames side by side
        combined = np.hstack([frame1, frame2])
        out.write(combined)
        
        frame_count += 1
    
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"✅ Comparison video created: {output_path}")


def main():
    """
    Main demo function.
    """
    print("🚀 Video Stabilization Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # File paths
    shaky_video = output_dir / "sample_shaky_video.mp4"
    
    # Step 1: Create sample shaky video
    create_sample_shaky_video(str(shaky_video), duration_seconds=8)
    
    # Step 2: Analyze original video motion
    print("\n📊 Analyzing original video motion...")
    original_motion = analyze_video_motion(str(shaky_video))
    print(f"Original motion analysis:")
    for key, value in original_motion.items():
        print(f"  {key}: {value:.2f}")
    
    # Progress callback function
    def progress_callback(current, total, stage):
        progress = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'\r{stage}: |{bar}| {progress:.1f}% ({current}/{total})', 
              end='', flush=True)
        if current == total:
            print()

    # --- Run demo for each algorithm ---
    algorithms_to_test = [
        {"name": "feature", "params": {"feature_detector": "ORB", "max_features": 1000}},
        {"name": "optical_flow", "params": {"max_features": 500}},
        {"name": "parametric", "params": {"parametric_motion_type": "homography"}}
    ]

    for algo_config in algorithms_to_test:
        algorithm = algo_config["name"]
        algo_specific_params = algo_config["params"]

        print("\n" + "=" * 50)
        print(f"🚀 Running demo for: {algorithm.upper()} ALGORITHM")
        print("=" * 50)

        # File paths
        stabilized_video = output_dir / f"stabilized_video_{algorithm}.mp4"
        comparison_video = output_dir / f"comparison_video_{algorithm}.mp4"
        metrics_file = output_dir / f"stabilization_metrics_{algorithm}.json"

        # Initialize and configure stabilizer
        print("\n⚙️ Initializing video stabilizer...")
        stabilizer_params = {
            "algorithm": algorithm,
            "smoothing_radius": 25,
            "border_mode": "black",
            "crop_ratio": 0.08,
            **algo_specific_params
        }
        stabilizer = VideoStabilizer(**stabilizer_params)

        # Perform stabilization
        print("\n🔧 Performing video stabilization...")
        start_time = time.time()
        success = stabilizer.stabilize_video(
            str(shaky_video),
            str(stabilized_video),
            progress_callback=progress_callback
        )
        processing_time = time.time() - start_time

        if not success:
            print(f"❌ Stabilization failed for {algorithm} algorithm!")
            continue

        print(f"✅ Stabilization completed in {processing_time:.1f} seconds")

        # Get stabilization metrics
        print("\n📈 Calculating stabilization metrics...")
        stabilization_metrics = stabilizer.get_stabilization_metrics()

        # Analyze stabilized video motion
        stabilized_motion = analyze_video_motion(str(stabilized_video))

        # Create comprehensive results
        results = {
            "processing_info": {
                "algorithm": algorithm,
                "processing_time_seconds": processing_time,
                **stabilizer_params
            },
            "original_motion_analysis": original_motion,
            "stabilized_motion_analysis": stabilized_motion,
            "stabilization_metrics": stabilization_metrics,
            "improvement_summary": {}
        }

        # Calculate improvements
        if original_motion and stabilized_motion:
            improvements = {
                "motion_reduction_x": (1 - stabilized_motion["std_motion_x"] / original_motion["std_motion_x"]) * 100,
                "motion_reduction_y": (1 - stabilized_motion["std_motion_y"] / original_motion["std_motion_y"]) * 100,
                "max_motion_reduction_x": (1 - stabilized_motion["max_motion_x"] / original_motion["max_motion_x"]) * 100,
                "max_motion_reduction_y": (1 - stabilized_motion["max_motion_y"] / original_motion["max_motion_y"]) * 100
            }
            results["improvement_summary"] = improvements

        # Save results
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)

        # Create comparison video
        create_comparison_video(str(shaky_video), str(stabilized_video), str(comparison_video))

        # Display results summary
        print("\n" + "-" * 50)
        print(f"📊 {algorithm.upper()} ALGORITHM RESULTS SUMMARY")
        print("-" * 50)
        print(f"\n📁 Output Files:")
        print(f"  • Stabilized video: {stabilized_video}")
        print(f"  • Comparison video: {comparison_video}")
        print(f"  • Detailed metrics: {metrics_file}")
        print(f"\n⏱️ Processing Performance:")
        print(f"  • Processing time: {processing_time:.1f} seconds")
        print(f"  • Frames processed: {original_motion.get('total_frames', 'N/A')}")
        if original_motion.get('total_frames'):
            fps_processed = original_motion['total_frames'] / processing_time
            print(f"  • Processing speed: {fps_processed:.1f} FPS")
        
        print(f"\n📉 Motion Reduction:")
        if 'improvement_summary' in results and results['improvement_summary']:
            improvements = results['improvement_summary']
            print(f"  • X-axis shake reduction: {improvements['motion_reduction_x']:.1f}%")
            print(f"  • Y-axis shake reduction: {improvements['motion_reduction_y']:.1f}%")
        
        print(f"\n🎯 Quality Metrics (from VideoStabilizer):")
        if stabilization_metrics:
            print(f"  • Motion reduction X: {stabilization_metrics.get('motion_reduction_x', 0):.1%}")
            print(f"  • Motion reduction Y: {stabilization_metrics.get('motion_reduction_y', 0):.1%}")
            print(f"  • Original variance X: {stabilization_metrics.get('original_variance_x', 0):.1f}")
            print(f"  • Smoothed variance X: {stabilization_metrics.get('smoothed_variance_x', 0):.1f}")

    print("\n🎉 All demos completed successfully!")


if __name__ == "__main__":
    main()


