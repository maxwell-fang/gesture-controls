import numpy as np
import torch

def normalize_keypoints(keypoints, img_sz):
    H, W = img_sz

    norm_keypoints = torch.zeros(keypoints.shape)

    norm_keypoints = keypoints / torch.tensor([H, W])

    return norm_keypoints