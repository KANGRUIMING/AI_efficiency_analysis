import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Test with a simple image
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Test with a blank image
image = cv2.imread('test_image.jpg') if cv2.imread('test_image.jpg') is not None else \
    cv2.imread('C:/Windows/Web/Wallpaper/Windows/img0.jpg')  # Use Windows default wallpaper as fallback

# Process the test image
if image is not None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    print("MediaPipe successfully processed the image!")
else:
    print("No test image found, but MediaPipe imported successfully.")

print("MediaPipe installation verified!") 