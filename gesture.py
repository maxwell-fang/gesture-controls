import numpy as np
import cv2
from pipeline import HandPosePipeline

class Gesture():
    def __init__(self, name, frame_skip, precision, expansion_coeff):

        self.name = name
        self.frame_skip = frame_skip
        self.keypoint_precision = precision
        self.exp_coeff = expansion_coeff

        self.keypoints = ''

    def __len__(self):
        return self.samples

    def create_keypoints(self, video: np.ndarray | str):

        if type(video) is str:
            cap = cv2.VideoCapture(video)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame_rgb)

            cap.release()

            video = np.stack(frames, axis=0)

        pipeline = HandPosePipeline('./models/hand_detector.pt', './models/pose_estimator.pt', 'cuda')

        C, H, W = frame_rgb.shape

        keypoints, coords = pipeline.process_frames(video)

