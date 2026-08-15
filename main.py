import torch
from gesture import Gesture
from gesture_clustering import GestureClustering, create_boolean_mask
from pipeline import HandPosePipeline
import cv2

def main(config):

    models = HandPosePipeline('./models/hand_detector.pt', './models/pose_estimator.pt', 'cuda')

    gesture_clustering = GestureClustering('gestures', 5, 0.2)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])

    # actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(f"Actual Resolution: {actual_width}x{actual_height}")

    while True:


if not cap.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Webcam accessed successfully!")
 
# Read the first frame to confirm capturing
ret, frame = cap.read()
 
if ret:
 
    # Display the frame using imshow
    cv2.imwrite('./test.png', frame)
else:
    print("Error: Could not capture a frame.")
 
# Release the webcam
cap.release()