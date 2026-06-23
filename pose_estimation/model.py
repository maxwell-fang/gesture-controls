from torch import nn
import torch

class HandJointsDetection(nn.Module):

    def __init__(self, img_size=224, embedding_dim=3):
        self.img_size = img_size
        self.embedding_dim = embedding_dim
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
        
        self.hourglass_lyrs = Hourglass(height=2, depth=2, channels=256, reduction_channels=128)

        self.pbbranches_lyrs = PartsBasedBranches(branch_groups=[1, 4, 4, 4, 4, 4], width=56)

        self.loss_fcn = HeatMapsMSELoss()

    def forward(self, x):
        y = x
        for lyr in self.conv_pool_lyrs:
            y = lyr(y)
        
        y = self.hourglass_lyrs(y)

        z = self.pbbranches_lyrs(y)

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
        self.loss_fcn = nn.MSELoss()

    def forward(self, inputs, targets, visibility):
        mask = visibility.view(visibility.size(0), visibility.size(1), 1, 1)

        mask = (mask > 0).float()

        loss = self.loss_fcn(inputs, targets)

        loss = loss * mask

        return loss.mean()