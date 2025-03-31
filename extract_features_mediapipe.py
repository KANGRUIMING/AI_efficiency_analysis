import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path

class MediaPipeFeatureExtractor:
    def __init__(self, 
                 videos_dir, 
                 keypoints_dir, 
                 visualizations_dir, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """Initialize the MediaPipe feature extractor."""
        self.videos_dir = videos_dir
        self.keypoints_dir = keypoints_dir
        self.visualizations_dir = visualizations_dir
        
        # Create directories if they don't exist
        Path(self.keypoints_dir).mkdir(parents=True, exist_ok=True)
        Path(self.visualizations_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        # Configuration
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
    def process_video(self, video_path, output_csv=None, create_visualization=True, max_frames=None):
        """
        Process a video with MediaPipe to extract pose and hand keypoints.
        
        Args:
            video_path: Path to the video file
            output_csv: Path to save the CSV data (if None, generates based on video name)
            create_visualization: Whether to create visualization video
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            DataFrame containing the extracted keypoints
        """
        print(f"Processing video: {video_path}")
        video_name = os.path.basename(video_path).split('.')[0]
        
        if output_csv is None:
            output_csv = os.path.join(self.keypoints_dir, f"{video_name}_keypoints.csv")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize MediaPipe Pose and Hands
        all_keypoints = []
        frame_idx = 0
        start_time = time.time()
        
        # Setup visualization writer if needed
        vis_writer = None
        if create_visualization:
            vis_path = os.path.join(self.visualizations_dir, f"{video_name}_visualization.mp4")
            vis_writer = cv2.VideoWriter(
                vis_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (frame_width, frame_height)
            )
        
        # Process frames with MediaPipe
        with self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence) as pose:
            
            with self.mp_hands.Hands(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence) as hands:
                
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break
                    
                    if max_frames is not None and frame_idx >= max_frames:
                        break
                    
                    # Convert to RGB for MediaPipe
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    
                    # Process with MediaPipe
                    pose_results = pose.process(image_rgb)
                    hands_results = hands.process(image_rgb)
                    
                    # Prepare for drawing
                    image_rgb.flags.writeable = True
                    image_viz = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Extract timestamp
                    timestamp = frame_idx / fps
                    
                    # Initialize keypoints for this frame
                    frame_data = {'frame': frame_idx, 'timestamp': timestamp}
                    
                    # Extract pose landmarks if detected
                    if pose_results.pose_landmarks:
                        # Draw pose landmarks
                        self.mp_drawing.draw_landmarks(
                            image_viz,
                            pose_results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        # Extract pose keypoints
                        for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            frame_data[f'pose_{i}_x'] = landmark.x
                            frame_data[f'pose_{i}_y'] = landmark.y
                            frame_data[f'pose_{i}_z'] = landmark.z
                            frame_data[f'pose_{i}_visibility'] = landmark.visibility
                    
                    # Extract hand landmarks if detected
                    if hands_results.multi_hand_landmarks:
                        for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                            # Draw hand landmarks
                            self.mp_drawing.draw_landmarks(
                                image_viz,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                            
                            # Determine if this is left or right hand
                            hand_type = "left" if hand_idx < len(hands_results.multi_handedness) and \
                                               hands_results.multi_handedness[hand_idx].classification[0].label == "Left" else "right"
                            
                            # Record hand type
                            frame_data[f'hand_{hand_idx}_type'] = 0 if hand_type == "left" else 1
                            
                            # Extract hand keypoints
                            for i, landmark in enumerate(hand_landmarks.landmark):
                                frame_data[f'hand_{hand_idx}_{i}_x'] = landmark.x
                                frame_data[f'hand_{hand_idx}_{i}_y'] = landmark.y
                                frame_data[f'hand_{hand_idx}_{i}_z'] = landmark.z
                    
                    # Add motion estimation
                    if frame_idx > 0 and 'pose_0_x' in frame_data and 'pose_0_x' in all_keypoints[-1]:
                        # Simple motion metric using key points
                        motion = 0
                        count = 0
                        
                        for i in range(33):  # MediaPipe Pose has 33 landmarks
                            if f'pose_{i}_x' in frame_data and f'pose_{i}_x' in all_keypoints[-1]:
                                dx = frame_data[f'pose_{i}_x'] - all_keypoints[-1][f'pose_{i}_x']
                                dy = frame_data[f'pose_{i}_y'] - all_keypoints[-1][f'pose_{i}_y']
                                dist = np.sqrt(dx**2 + dy**2)
                                motion += dist
                                count += 1
                        
                        if count > 0:
                            frame_data['avg_motion'] = motion / count
                    else:
                        frame_data['avg_motion'] = 0
                    
                    all_keypoints.append(frame_data)
                    
                    # Write visualization frame
                    if vis_writer is not None:
                        # Add frame number to visualization
                        cv2.putText(image_viz, f"Frame: {frame_idx}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        vis_writer.write(image_viz)
                    
                    frame_idx += 1
                    if frame_idx % 30 == 0:
                        print(f"Processed {frame_idx} frames...")
        
        # Release resources
        cap.release()
        if vis_writer is not None:
            vis_writer.release()
        
        # Create DataFrame and save to CSV
        if all_keypoints:
            df = pd.DataFrame(all_keypoints)
            df.to_csv(output_csv, index=False)
            print(f"Saved keypoints to {output_csv}")
            print(f"Processed {frame_idx} frames in {time.time() - start_time:.2f} seconds")
            return df
        else:
            print("No keypoints were extracted from the video.")
            return None
    
    def process_all_videos(self, extensions=('.mp4', '.avi', '.mov'), max_frames=None):
        """Process all videos in the videos directory with supported extensions."""
        videos = [f for f in os.listdir(self.videos_dir) 
                 if any(f.lower().endswith(ext) for ext in extensions)]
        
        if not videos:
            print(f"No videos found in {self.videos_dir} with extensions {extensions}")
            return
        
        print(f"Found {len(videos)} videos to process")
        
        for video_file in videos:
            video_path = os.path.join(self.videos_dir, video_file)
            self.process_video(video_path, max_frames=max_frames)

def extract_features(video_path, output_csv=None, use_mediapipe=True, force_mediapipe=False, 
                      keypoints_dir=None, visualizations_dir=None):
    """
    Extract features from a video using MediaPipe.
    
    This is a compatibility function to match the interface expected by analyze_efficiency.py.
    
    Args:
        video_path: Path to the video
        output_csv: Path to save CSV data
        use_mediapipe: Whether to use MediaPipe (should be True)
        force_mediapipe: Force using MediaPipe even if previous extraction exists
        keypoints_dir: Directory to save keypoints data
        visualizations_dir: Directory to save visualizations
    
    Returns:
        DataFrame with extracted keypoints
    """
    if not use_mediapipe:
        print("WARNING: MediaPipe is required for this script. Setting use_mediapipe=True.")
    
    # Use provided directories or fall back to defaults
    keypoints_dir = keypoints_dir or r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\keypoints_data"
    visualizations_dir = visualizations_dir or r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\visualizations"
    
    # Check if output CSV already exists and we're not forcing re-extraction
    if output_csv and os.path.exists(output_csv) and not force_mediapipe:
        print(f"Using existing keypoints from {output_csv}")
        return pd.read_csv(output_csv)
    
    # Get parent directory for video if not using keypoints_dir
    video_dir = os.path.dirname(video_path) if keypoints_dir is None else None
    
    # Initialize extractor
    extractor = MediaPipeFeatureExtractor(
        videos_dir=os.path.dirname(video_path),
        keypoints_dir=keypoints_dir,
        visualizations_dir=visualizations_dir
    )
    
    # Process the video
    return extractor.process_video(video_path, output_csv)

if __name__ == "__main__":
    # Paths from your specifications
    videos_dir = r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\efficiency-selected"
    keypoints_dir = r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\keypoints_data"
    visualizations_dir = r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\visualizations"
    
    # Create extractor
    extractor = MediaPipeFeatureExtractor(
        videos_dir=videos_dir,
        keypoints_dir=keypoints_dir,
        visualizations_dir=visualizations_dir
    )
    
    # Process all videos
    print("Starting to process all videos...")
    extractor.process_all_videos()
    print("Completed processing all videos!")
    print(f"Keypoints saved to: {keypoints_dir}")
    print(f"Visualizations saved to: {visualizations_dir}")
    
    print("\nNext steps:")
    print("1. Run train_model.py to train an efficiency model with your keypoints data")
    print("2. Run analyze_efficiency.py to analyze specific videos") 