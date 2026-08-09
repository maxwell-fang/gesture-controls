import numpy as np
import cv2
from pipeline import HandPosePipeline
from pose_estimation import pred_to_original_size
from utils import normalize_keypoints
import torch
import pickle as pkl
import os

class Gesture():
    def __init__(self, name, frame_skip, precision, expansion_coeff):

        self.name = name
        self.frame_skip = frame_skip
        self.keypoint_precision = precision
        self.exp_coeff = expansion_coeff

        self.keypoints = np.empty(1)

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

        keypoints, coords = pipeline.process_frames(self.frame_skip, video, self.exp_coeff)

        bbox_sizes = coords[:, 0] - coords[:, 2]

        unnorm_keypoints = keypoints*bbox_sizes
        unnorm_keypoints = unnorm_keypoints + coords[:, :2]

        origin_shift = bbox_sizes[0, 0]*torch.ones([1, 2]) + coords[0, :2]

        norm_keypoints = normalize_keypoints(unnorm_keypoints - origin_shift, (H, W))

        movement_keypoints = torch.zeros_like(norm_keypoints)
        movement_keypoints[1:, :, :] = norm_keypoints[:-1, :, :]

        final_keypoints = norm_keypoints - movement_keypoints

        self.keypoints = final_keypoints

def save_gesture(gesture):

    if not os.path.exists('gestures'):
        os.mkdir('gestures')
        
    with open(f'./gestures/{gesture.name}.pkl', 'wb') as f:
        pkl.dump(gesture, f)

def load_gesture(gesture_path):

    gesture = Gesture('', 0, 0, 0, np.empty(1))

    with open(gesture_path, 'rb') as f:
        loaded_gesture = pkl.load(f)

    gesture.name = loaded_gesture.name
    gesture.frame_skip = loaded_gesture.frame_skip
    gesture.keypoint_precision = loaded_gesture.keypoint_precision
    gesture.exp_coeff = loaded_gesture.exp_coeff
    gesture.keypoints = loaded_gesture.keypoints

    return gesture
