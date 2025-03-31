import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Pose and Hands solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

def extract_keypoints(video_path, output_csv):
    """Extract pose and hand keypoints from a video and save to CSV."""
    cap = cv2.VideoCapture(video_path)
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    keypoints_data = []
    frame_num = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Process only every 3rd frame to reduce computation
        if frame_num % 3 == 0:
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get pose keypoints
            pose_results = pose.process(image_rgb)
            
            # Get hand keypoints
            hands_results = hands.process(image_rgb)
            
            # Process and store results
            timestamp = frame_num / fps
            
            frame_data = {'frame': frame_num, 'timestamp': timestamp}
            
            # Process pose data
            if pose_results.pose_landmarks:
                for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    frame_data[f'pose_{i}_x'] = landmark.x
                    frame_data[f'pose_{i}_y'] = landmark.y
                    frame_data[f'pose_{i}_z'] = landmark.z
                    frame_data[f'pose_{i}_visibility'] = landmark.visibility
            
            # Process hands data
            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        frame_data[f'hand_{hand_idx}_{i}_x'] = landmark.x
                        frame_data[f'hand_{hand_idx}_{i}_y'] = landmark.y
                        frame_data[f'hand_{hand_idx}_{i}_z'] = landmark.z
            
            keypoints_data.append(frame_data)
        
        frame_num += 1
    
    cap.release()
    
    # Save to CSV
    df = pd.DataFrame(keypoints_data)
    df.to_csv(output_csv, index=False)
    print(f"Extracted keypoints saved to {output_csv}")
    return df

def process_videos(video_dir, output_dir):
    """Process all videos in a directory and extract keypoints."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(video_dir):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_keypoints.csv")
            extract_keypoints(video_path, output_path)

if __name__ == "__main__":
    # Example usage
    process_videos("efficiency-selected", "keypoints_data") 