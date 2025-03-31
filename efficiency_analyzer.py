import cv2
import numpy as np
import csv

# Define video path
video_path = r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\efficiency-selected\IMG_3925.MOV"

# Function to play the video at regular speed and allow labeling using keyboard keys
def label_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame rate (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # List to store the frame labels
    frame_labels = []  # List to store (frame number, label)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow("Video", frame)
        
        # Wait for a key press, ensuring the video plays at regular speed
        key = cv2.waitKey(int(1000 / fps)) & 0xFF  # waits for the correct time based on FPS
        
        # If the user presses a key ('1', '2', '3', etc.), record the label and frame number
        if key == ord('1'):  # Press '1' to label Step 1
            frame_labels.append((cap.get(cv2.CAP_PROP_POS_FRAMES), 'Step 1'))
        elif key == ord('2'):  # Press '2' to label Step 2
            frame_labels.append((cap.get(cv2.CAP_PROP_POS_FRAMES), 'Step 2'))
        elif key == ord('3'):  # Press '3' to label Step 3
            frame_labels.append((cap.get(cv2.CAP_PROP_POS_FRAMES), 'Step 3'))
        elif key == ord('4'):  # Press '4' to label Step 4
            frame_labels.append((cap.get(cv2.CAP_PROP_POS_FRAMES), 'Step 4'))
        elif key == ord('5'):  # Press '5' to label Step 5
            frame_labels.append((cap.get(cv2.CAP_PROP_POS_FRAMES), 'Step 5'))
        elif key == ord('6'):  # Press '6' to label Step 6
            frame_labels.append((cap.get(cv2.CAP_PROP_POS_FRAMES), 'Step 6'))
        elif key == ord('q'):  # Press 'q' to quit labeling
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return frame_labels

# Function to save the labels to a CSV file
def save_labels_to_csv(frame_labels, output_file="labeled_steps.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Step Label"])  # Column headers
        for frame_num, label in frame_labels:
            writer.writerow([int(frame_num), label])

# Call the function to label the video
frame_labels = label_video(video_path)

# Print the frame labels (frame number and associated label)
for frame_num, label in frame_labels:
    print(f"Frame {int(frame_num)}: {label}")

# Save the labeled steps to a CSV file
save_labels_to_csv(frame_labels)
