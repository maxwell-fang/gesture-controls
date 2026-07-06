import cv2
from ultralytics import YOLO
import numpy as np

def pad_bb_square(frame, bb_coords):
    frame_h, frame_w, __ = frame.shape

    # assumes coords are in (top left x, top left y, bottom right x, bottom right y)
    w = int(bb_coords[2]) - int(bb_coords[0])
    h = int(bb_coords[3]) - int(bb_coords[1])

    if w > h:
        pad_dist = (w - h)/2
        new_coords = [bb_coords[0], round(bb_coords[1] - pad_dist), bb_coords[2], round(bb_coords[1] + pad_dist)]

    if w < h:
        pad_dist = (h - w)/2
        new_coords = [round(bb_coords[0] - pad_dist), bb_coords[1], round(bb_coords[2] - pad_dist), bb_coords[3]]

    if new_coords[0] < 0:
        new_coords[2] = new_coords[2] - new_coords[0]

    if new_coords[1] < 0:
        new_coords[3] = new_coords[3] - new_coords[1]

    if new_coords[2] > frame_w:
        new_coords[0] = new_coords[0] - new_coords[2] + frame_w

    if new_coords[3] > frame_w:
        new_coords[1] = new_coords[1] - new_coords[3] + frame_w
        
    new_image = frame[new_coords[0]:new_coords[2], new_coords[1]:new_coords[3], :]

    return new_image, new_coords