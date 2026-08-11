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

    while True:
