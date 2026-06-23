import torch
from torchvision.datasets import CocoDetection

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
    visibility = keypoints[:, 2].view(N)
    
    dist_sq = (grid_x - target_x) ** 2 + (grid_y - target_y) ** 2
    
    heatmaps = torch.exp(-dist_sq / (2 * (std ** 2)))
    
    return heatmaps, visibility

class HandsDataset(CocoDetection):
    def __init__(self, root='', annFile='', transform=None, dataset_name='dataset', heatmaps=True):
        super().__init__(root=root, annFile=annFile)
        self.name = dataset_name
        self.transform = transform
        print(f'{dataset_name} Loaded.')
        self.joints_map = self.coco.cats.get(1)['skeleton']
        self.heatmaps = heatmaps

    def __getitem__(self, index):
        image, metadata = super().__getitem__(index)

        keypoints = torch.tensor(metadata[0]['keypoints']).view(21, 3).float()
        if self.transform:
            image = self.transform(image)
        if self.heatmaps:
            target_heatmaps, visibility = generate_heatmaps(imgsize=(224, 224), keypoints=keypoints, std=1.0, downscale_factor=4)
            return image, (target_heatmaps, visibility)

        else:
            return image, keypoints