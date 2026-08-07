from ultralytics import YOLO
from pose_estimation import load_model, HandJointsDetection
from hand_detection import expand_bbox_for_wrist, pad_bb_square
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

class HandPosePipeline():
    def __init__(self, hand_det_path, pose_est_path, device):
        self.detector = YOLO(hand_det_path)
        self.detector.to(device)
        self.pose_estimator = load_model(pose_est_path)
        self.pose_estimator.to(device)
        self.device = device

    def process_frame(self, frame, exp_coeff=0.4):
        results = self.detector.predict(frame, verbose=False)[0]
        boxes = results.boxes[0]
        b = boxes.xyxy[0]
        b = expand_bbox_for_wrist(b, frame.shape, exp_coeff)
        hand_image, coords = pad_bb_square(frame, b)

        input_image = TF.to_tensor(cv2.resize(hand_image, self.pose_estimator.img_size))
        keypoints = self.pose_estimator.predict(input_image)

        return keypoints, torch.tensor(coords)

    def process_frames(self, frame_skip, frames, exp_coeff=0.4):

        N, H, W, C = frames.shape

        input_frames = frames[::frame_skip, :, :, :]
        hand_frames = torch.zeros((N // frame_skip, self.pose_estimator.img_size, self.self.pose_estimator.img_size, C))
        bbox_coords = []

        results = self.detector.predict(input_frames, verbose=False)

        for i, r in enumerate(results):
            frame = input_frames[i, :, :, :]
            boxes = r.boxes[0]
            b = boxes.xyxy[0]
            b = expand_bbox_for_wrist(b, (H, W, C), exp_coeff)
            hand_image, coords = pad_bb_square(frame, b)
            bbox_coords.append(coords)
            hand_frames[i, :, :, :] = TF.to_tensor(cv2.resize(hand_image, self.pose_estimator.img_size))

        keypoints = self.pose_estimator.predict(hand_frames)

        return keypoints, torch.tensor(bbox_coords)