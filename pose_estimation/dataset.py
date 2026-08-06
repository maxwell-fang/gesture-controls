import torch
import os
import json
import cv2
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
import torchvision.transforms as T
from glob import glob

def generate_heatmaps(imgsize=(224,224), keypoints=torch.Tensor, std=1.0, downscale_factor=4):

    N, _ = keypoints.shape
    H, W = imgsize
    device = keypoints.device
    heatmap_h = H // downscale_factor
    heatmap_w = W // downscale_factor
    
    grid_y, grid_x = torch.meshgrid(
        torch.arange(heatmap_h, device=device, dtype=torch.float32),
        torch.arange(heatmap_w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    grid_x = grid_x.view(1, heatmap_h, heatmap_w)
    grid_y = grid_y.view(1, heatmap_h, heatmap_w)
    
    target_x = (keypoints[:, 0] / downscale_factor).view(N, 1, 1)
    target_y = (keypoints[:, 1] / downscale_factor).view(N, 1, 1)
    # visibility = keypoints[:, 2].view(N)
    
    dist_sq = (grid_x - target_x) ** 2 + (grid_y - target_y) ** 2
    
    heatmaps = torch.exp(-dist_sq / (2 * (std ** 2)))
    
    # return heatmaps, visibility
    return heatmaps

class HanCoDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (str): Path containing 'rgb', 'annotations', and 'calibration' folders.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root
        self.rgb_dir = os.path.join(root, 'HanCo_rgb_color_auto', 'rgb_color_auto')
        self.annot_dir = os.path.join(root, 'HanCo_xyz', 'xyz')
        self.calib_dir = os.path.join(root, 'HanCo_calib_meta', 'calib')
        self.transform = transform if transform else T.ToTensor()

        # Step 1: Pre-index all (video_id, camera_id, frame_id) tuples
        self.samples = []
        self._index_dataset()

    def _index_dataset(self):
        """Discovers all valid image frames across videos and cameras."""
        video_folders = sorted(os.listdir(self.rgb_dir))
        for video_id in video_folders:
            video_path = os.path.join(self.rgb_dir, video_id)
            if not os.path.isdir(video_path):
                continue
                
            camera_folders = sorted(os.listdir(video_path))
            for cam_id in camera_folders:
                cam_path = os.path.join(video_path, cam_id)
                cam_no = cam_id[3]
                if not os.path.isdir(cam_path):
                    continue

                frame_files = sorted(os.listdir(cam_path))
                for frame_file in frame_files:
                    if frame_file.endswith(('.jpg', '.png')):
                        frame_id = os.path.splitext(frame_file)[0]
                        self.samples.append({
                            'video_id': video_id,
                            'cam_id': cam_no,
                            'frame_id': frame_id,
                            'img_path': os.path.join(cam_path, frame_file),
                            'annotation_path': os.path.join(self.annot_dir, video_id, frame_id + '.json'),
                            'calibration_path': os.path.join(self.calib_dir, video_id, frame_id + '.json')
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        cam_id = sample_info['cam_id']
        with open(sample_info['annotation_path'], 'r') as f:
            sample_annot = json.load(f)

        with open(sample_info['calibration_path'], 'r') as f:
            sample_calib = json.load(f)

        image = cv2.imread(sample_info['img_path'])
        if image is None:
            raise FileNotFoundError(f"Could not load image at {sample_info['img_path']}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = image.shape[:2]

        if self.transform:
            image = self.transform(image)

        keypoints_3d = torch.tensor(sample_annot, dtype=torch.float32)

        raw_k = sample_calib['K'][int(cam_id)]
        raw_m = sample_calib['M'][int(cam_id)]
        K = torch.tensor(raw_k, dtype=torch.float32)
        M = torch.tensor(raw_m, dtype=torch.float32)

        R = M[:3, :3]
        t = M[:3, 3]

        keypoints_3d_cam = (keypoints_3d @ R.T) + t

        X = keypoints_3d_cam[:, 0]
        Y = keypoints_3d_cam[:, 1]
        Z = torch.clamp(keypoints_3d_cam[:, 2], min=1e-5)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_pixel = (X * fx / Z) + cx
        y_pixel = (Y * fy / Z) + cy
        x_norm = x_pixel / orig_w
        y_norm = y_pixel / orig_h

        keypoints_2d_norm = torch.stack([x_norm, y_norm], dim=-1)

        in_bounds_x = (x_norm >= 0.0) & (x_norm <= 1.0)
        in_bounds_y = (y_norm >= 0.0) & (y_norm <= 1.0)
        valid_depth = (Z > 0.0)
        visibility = (in_bounds_x & in_bounds_y & valid_depth).float()

        target_heatmaps = generate_heatmaps(imgsize=(224,224), keypoints=torch.stack([x_pixel, y_pixel, visibility], dim=-1), std=2.0, downscale_factor=4)

        return {
            'image': image,                         # Transformed tensor
            'target_heatmaps': target_heatmaps,
            'image_path': sample_info['img_path'],
            'keypoints_2d_norm': keypoints_2d_norm, # [21, 2] in [0, 1] for Soft-Argmax loss
            'visibility': visibility,               # [21] mask
            'keypoints_3d': keypoints_3d,           # [21, 3] camera space coords
            'cam_matrix': K          # [3, 4] padded matrix for batching
        }

class FreiHand(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (str): Path containing 'rgb', 'annotations', and 'calibration' folders.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root
        self.rgb_dir = os.path.join(root, 'evaluation', 'rgb')
        self.transform = transform if transform else T.ToTensor()

        # Step 2: Cache annotation files in memory (one JSON per video)
        with open(os.path.join(self.root_dir, 'evaluation_xyz.json'), 'r') as f:
            self.annotations = json.load(f)

        with open(os.path.join(self.root_dir, 'evaluation_K.json'), 'r') as f:
            self.calibrations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = os.path.join(self.rgb_dir, f"{idx:08d}.jpg")

        image = cv2.imread(os.path.join(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image at {os.path.join(image_path)}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = image.shape[:2]

        if self.transform:
            image = self.transform(image)

        keypoints_3d = torch.tensor(self.annotations[idx], dtype=torch.float32)
        K = torch.tensor(self.calibrations[idx], dtype=torch.float32)

        X = keypoints_3d[:, 0]
        Y = keypoints_3d[:, 1]
        Z = torch.clamp(keypoints_3d[:, 2], min=1e-5)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_pixel = (X * fx / Z) + cx
        y_pixel = (Y * fy / Z) + cy
        x_norm = x_pixel / orig_w
        y_norm = y_pixel / orig_h

        keypoints_2d_norm = torch.stack([x_norm, y_norm], dim=-1)

        in_bounds_x = (x_norm >= 0.0) & (x_norm <= 1.0)
        in_bounds_y = (y_norm >= 0.0) & (y_norm <= 1.0)
        valid_depth = (Z > 0.0)
        visibility = (in_bounds_x & in_bounds_y & valid_depth).float()

        target_heatmaps = generate_heatmaps(imgsize=(224,224), keypoints=torch.stack([x_pixel, y_pixel, visibility], dim=-1), std=2.0, downscale_factor=4)

        return {
            'image': image,                         # Transformed tensor
            'image_path': image_path,
            'target_heatmaps': target_heatmaps,
            'keypoints_2d_norm': keypoints_2d_norm, # [21, 2] in [0, 1] for Soft-Argmax loss
            'visibility': visibility,               # [21] mask
            'keypoints_3d': keypoints_3d,           # [21, 3] camera space coords
            'cam_matrix': K          # [3, 4] padded matrix for batching
        }

