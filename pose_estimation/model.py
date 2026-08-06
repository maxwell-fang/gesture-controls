from torch import nn
import torch

class HandJointsDetection(nn.Module):

    def __init__(self,joint_map: list, img_size=224, embedding_dim=3):
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.joint_map = joint_map
        super().__init__()

        self.conv_pool_lyrs = nn.ModuleList()
        self.conv_pool_lyrs.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1))
        self.conv_pool_lyrs.append(nn.BatchNorm2d(num_features=64))
        self.conv_pool_lyrs.append(nn.LeakyReLU())
        self.conv_pool_lyrs.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_pool_lyrs.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        self.conv_pool_lyrs.append(nn.BatchNorm2d(num_features=64))
        self.conv_pool_lyrs.append(nn.LeakyReLU())
        self.conv_pool_lyrs.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        self.conv_pool_lyrs.append(nn.BatchNorm2d(num_features=64))
        self.conv_pool_lyrs.append(nn.LeakyReLU())
        self.conv_pool_lyrs.append(ResBottleneck(reduction_channels=32, in_channels=64, out_channels=128))
        self.conv_pool_lyrs.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_pool_lyrs.append(ResBottleneck(reduction_channels=64, in_channels=128, out_channels=128))
        self.conv_pool_lyrs.append(ResBottleneck(reduction_channels=128, in_channels=128, out_channels=256))
        self.bottleneck_drop = nn.Dropout2d(p=0.2)
        
        self.hourglass_lyrs = Hourglass(height=2, depth=2, channels=256, reduction_channels=128)
        self.hourglass_drop = nn.Dropout2d(p=0.15)

        self.pbbranches_lyrs = PartsBasedBranches(branch_groups=[1, 4, 4, 4, 4, 4], width=56)

        loss_fcns = [HeatMapsMSELoss()]
        loss_weights = [1.0]

        self.loss_fcn = CombinedLoss(loss_fcns=loss_fcns, weights=loss_weights)

    def forward(self, x):
        y = x
        for lyr in self.conv_pool_lyrs:
            y = lyr(y)

        y = self.bottleneck_drop(y)
        
        y = self.hourglass_lyrs(y)

        y = self.hourglass_drop(y)

        y = self.pbbranches_lyrs(y)

        return y

class ResBottleneck(nn.Module):

    def __init__(self, reduction_channels=64, in_channels=128, out_channels=256):
        self.reduction_channels = reduction_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()

        self.res_lyrs = nn.ModuleList()
        self.res_lyrs.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduction_channels, kernel_size=1))
        self.res_lyrs.append(nn.BatchNorm2d(num_features=self.reduction_channels))
        self.res_lyrs.append(nn.LeakyReLU())
        self.res_lyrs.append(nn.Conv2d(in_channels=self.reduction_channels, out_channels=self.reduction_channels, kernel_size=3, padding=1))
        self.res_lyrs.append(nn.BatchNorm2d(num_features=self.reduction_channels))
        self.res_lyrs.append(nn.LeakyReLU())
        self.res_lyrs.append(nn.Conv2d(in_channels=self.reduction_channels, out_channels=self.out_channels, kernel_size=1))
        self.res_lyrs.append(nn.BatchNorm2d(num_features=self.out_channels))
        self.res_lyrs.append(nn.LeakyReLU())

        self.skip_lyr = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)

        self.final_lyr = nn.LeakyReLU()

    def forward(self, x):
        y = x

        for lyr in self.res_lyrs:
            y = lyr(y)

        skip_out = self.skip_lyr(x)

        y = self.final_lyr(y + skip_out)

        return y
        
class Hourglass(nn.Module):

    def __init__(self, height=3, depth=3, channels=256, reduction_channels=128):
        self.height = height
        self.depth = depth
        self.channels = channels
        self.reduction_channels = reduction_channels
        super().__init__()

        self.residual_bottleneck = ResBottleneck(reduction_channels=self.reduction_channels, in_channels=self.channels, out_channels=self.channels)
        self.batchnorm = nn.BatchNorm2d(num_features=self.channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act_fun = nn.LeakyReLU()

        self.up_res = ResBottleneck(reduction_channels=self.reduction_channels, in_channels=self.channels, out_channels=self.channels)

        self.upsample = nn.Upsample(scale_factor=2)

        if depth > 0:
            self.inner_hourglass = Hourglass(height=self.height, depth=self.depth - 1, channels=self.channels, reduction_channels=self.channels)
        else:
            self.inner_hourglass = None

    def forward(self, x):
        y = x

        y = self.residual_bottleneck(y)
        y_init = y

        y = self.batchnorm(y)
        y = self.act_fun(y)
        y = self.pool(y)

        if self.depth > 0:
            y = self.inner_hourglass(y)

        y = self.upsample(y)
        y = y_init + y

        return y
    
class PartsBasedBranches(nn.Module):

    def __init__(self, branch_groups=[1], width=56):
        self.no_branches = len(branch_groups)
        self.width = width
        super().__init__()

        self.branches = nn.ModuleList()

        for branch in range(self.no_branches):

            branch_lyrs = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            ResBottleneck(reduction_channels=32, in_channels=64, out_channels=64),
            nn.Conv2d(in_channels=64, out_channels=branch_groups[branch], kernel_size=1),
            nn.BatchNorm2d(num_features=branch_groups[branch]),
            nn.LeakyReLU())

            self.branches.append(branch_lyrs)

    def forward(self, x):
        y = []
        
        for branch in self.branches:
            branch_out = branch(x)
            y.append(branch_out)
 
        return torch.cat(y, dim=1)
    
class HeatMapsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_type = 'heatmap'
        self.loss_fcn = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets, visibility):
        mask = visibility.view(visibility.size(0), visibility.size(1), 1, 1)

        mask = (mask > 0).float()

        loss = self.loss_fcn(inputs, targets)

        loss = loss * mask
        return loss.sum() / (mask.sum() * targets.size(2) * targets.size(3) + 1e-8)
    
class SpatialConsistencyLoss(nn.Module):
    def __init__(self, joint_connections, temperature, image_size):
        super().__init__()
        self.target_type = 'keypoint'
        self.connections = joint_connections
        self.temperature = temperature
        self.image_size = image_size

    def forward(self, inputs, target_keypoints, visibility):
        """
        Args:
            pred_coords:   [Batch, Joints, 2] (Output from your Soft-Argmax)
            target_coords: [Batch, Joints, 2] (Ground truth coordinates)
            visibility:    [Batch, Joints]    (Binary/Float tracker: 1=visible, 0=invisible)
        """
        loss = 0.0
        valid_connections_count = 0

        pred_coords = _soft_argmax_2d(inputs, self.temperature)

        target_coords_norm = target_keypoints
        
        for joint_a, joint_b in self.connections:
            # 1. A bone is only valid if BOTH joints are visible
            # visibility[:, joint_a] -> [Batch]
            bone_visibility_mask = (visibility[:, joint_a] > 0) & (visibility[:, joint_b] > 0)
            bone_visibility_mask = bone_visibility_mask.float() # Convert boolean to 1.0 or 0.0

            # 2. Calculate the predicted and ground-truth bone lengths (Euclidean distance)
            pred_dist = torch.norm(pred_coords[:, joint_a] - pred_coords[:, joint_b], dim=-1)
            target_dist = torch.norm(target_coords_norm[:, joint_a] - target_coords_norm[:, joint_b], dim=-1)
            
            # 3. Calculate squared error for this specific bone connection across the batch
            connection_loss = nn.functional.smooth_l1_loss(pred_dist, target_dist, reduction='none')
            
            # 4. Mask the loss so invisible bones contribute 0.0 to the gradient
            masked_connection_loss = connection_loss * bone_visibility_mask
            
            # Accumulate
            loss += masked_connection_loss.sum()
            valid_connections_count += bone_visibility_mask.sum()

        # 5. Normalize by the total number of valid, visible bones across the entire batch
        # Add a tiny epsilon (1e-8) to prevent division by zero if an entire batch is occluded
        return loss / (valid_connections_count + 1e-8)
    
class MultimodalPenalty(nn.Module):
    def __init__(self, loss_fcn, temperature, image_size):
        super().__init__()
        self.target_type = 'keypoint'
        self.loss_fcn = loss_fcn
        self.temperature = temperature
        self.image_size = image_size

    def forward(self, inputs, target_keypoints, visibility):

        pred_coords = _soft_argmax_2d(inputs, self.temperature)
        target_coords_norm = target_keypoints
        raw_loss = self.loss_fcn(pred_coords, target_coords_norm, reduction='none')
        masked_loss = raw_loss.sum(dim=-1) * (visibility > 0).float()
        loss = masked_loss.sum() / ((visibility > 0).sum() + 1e-8)
        return loss
    
class CombinedLoss(nn.Module):
    def __init__(self, loss_fcns, weights):
        super().__init__()
        self.loss_fcns = loss_fcns
        self.weights = weights
    
    def forward(self, inputs, target_keypoints, target_heatmaps, visibility):

        loss = 0.0

        for loss_fcn, weight in zip(self.loss_fcns, self.weights):

            if loss_fcn.target_type == 'keypoint':
                loss += weight*loss_fcn(inputs, target_keypoints, visibility)
            elif loss_fcn.target_type == 'heatmap':
                loss += weight*loss_fcn(inputs, target_heatmaps, visibility)

        return loss

def _soft_argmax_2d(heatmaps, temperature):
    """
    Converts 4D heatmaps [B, C, H, W] into 2D coordinates [B, C, 2].
    """
    B, C, H, W = heatmaps.shape
    device = heatmaps.device

    scaled_heatmaps = heatmaps/temperature

    # 1. Apply spatial softmax to convert heatmaps to clean probability distributions
    # Flatten H and W into a single dimension to apply softmax across the grid
    probs = nn.functional.softmax(scaled_heatmaps.view(B, C, H * W), dim=-1).view(B, C, H, W)

    # 2. Create coordinates grids
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, steps=H, device=device, dtype=torch.float32),
        torch.linspace(0.0, 1.0, steps=W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # 3. Compute expected values (centers of mass)
    # Target shape for coordinates: [B, C, 2] -> (x, y)
    pred_x = torch.sum(probs * grid_x, dim=(-2, -1))
    pred_y = torch.sum(probs * grid_y, dim=(-2, -1))

    return torch.stack([pred_x, pred_y], dim=-1)

def hard_argmax_2d(heatmaps):
    """
    Extracts exact peak (x, y) coordinates in normalized [0, 1] range.
    """
    B, C, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, C, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    
    pred_y = (max_indices // W).float() / (H - 1)
    pred_x = (max_indices % W).float() / (W - 1)
    
    return torch.stack([pred_x, pred_y], dim=-1)