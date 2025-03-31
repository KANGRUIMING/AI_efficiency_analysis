import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to process video and extract pose keypoints
def extract_pose_keypoints(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # List to store the keypoints for each frame
    keypoints_list = []
    
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB (MediaPipe uses RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get the pose landmarks
        results = pose.process(frame_rgb)
        
        # Check if landmarks are detected
        if results.pose_landmarks:
            # List to store keypoints for this frame
            keypoints = []
            
            # Extract the pose landmarks (each landmark has x, y, z, and visibility)
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # Append the keypoints of this frame to the list
            keypoints_list.append(keypoints)
        else:
            # If no pose landmarks are detected, append None (optional, for error handling)
            keypoints_list.append(None)
    
    # Release the video capture object
    cap.release()
    
    # Return the list of keypoints for all frames
    return keypoints_list

# Function to save the keypoints to a CSV file
def save_keypoints_to_csv(keypoints, output_file="keypoints.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header with column names
        header = []
        for i in range(33):  # 33 pose landmarks
            header.extend([f"x{i}", f"y{i}", f"z{i}", f"visibility{i}"])  # Column for each landmark
        writer.writerow(header)
        
        # Write the keypoints for each frame
        for frame_keypoints in keypoints:
            if frame_keypoints is not None:  # Skip frames where no keypoints are detected
                flattened_keypoints = [value for landmark in frame_keypoints for value in landmark]
                writer.writerow(flattened_keypoints)
            else:
                # In case no landmarks detected in the frame, you can skip or write empty values
                writer.writerow([None] * 132)  # 33 landmarks * 4 values (x, y, z, visibility)

# Example usage of the function to extract keypoints from a video and save to CSV
video_path = r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\efficiency-selected\IMG_3925.MOV"
keypoints = extract_pose_keypoints(video_path)

# Save the extracted keypoints to a CSV file
save_keypoints_to_csv(keypoints, output_file="keypoints.csv")

print("Keypoints saved to 'keypoints.csv'")
