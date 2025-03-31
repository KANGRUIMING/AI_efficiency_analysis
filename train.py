import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Function to flatten keypoints
def flatten_keypoints(keypoints):
    return [value for landmark in keypoints for value in landmark] if keypoints is not None else [None] * 132  # 33 landmarks * 4 values (x, y, z, visibility)

# Example usage to extract keypoints from a video
video_path = r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\efficiency-selected\IMG_3925.MOV"
keypoints = extract_pose_keypoints(video_path)

# Print the number of frames and verify
print(f"Total frames in the video: {len(keypoints)}")

# Label data (Frame number, Step label)
frame_labels = [
    (31, 'Step 1'),
    (71, 'Step 2'),
    (102, 'Step 3'),
    (153, 'Step 4'),
    (230, 'Step 5'),
    (314, 'Step 6')
]

# Check if frame numbers exceed the total number of frames
for frame_num, label in frame_labels:
    if frame_num >= len(keypoints):
        print(f"Warning: Frame number {frame_num} is out of range! Total frames are {len(keypoints)}.")

# Map frame numbers to their labels
frame_to_label = dict(frame_labels)

# Prepare data for training
X = []  # Features (flattened keypoints)
y = []  # Labels (step labels)

# Loop through frame labels and extract corresponding keypoints
for frame_num, label in frame_labels:
    if frame_num < len(keypoints):  # Check if the frame number is within the range
        keypoints = keypoints[frame_num]  # Frame index corresponds to frame number in the list
        X.append(flatten_keypoints(keypoints))  # Flatten keypoints and append to features list
        y.append(label)  # Append the corresponding label (step)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier model
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Optionally, print predicted labels for the test set
for i, prediction in enumerate(y_pred):
    print(f"Test Frame {i + 1}: Predicted Step - {prediction}")
