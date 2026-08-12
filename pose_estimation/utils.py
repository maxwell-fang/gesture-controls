from pose_estimation import HandJointsDetection, HanCoDataset, FreiHand
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
import os
import cv2

# JOINTS_MAP = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
#                [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17],
#                  [17, 18], [18, 19], [19, 20]]

JOINTS_MAP = [
    # Thumb (1-4)
    [0, 1], [1, 2], [2, 3], [3, 4],
    # Index (5-8)
    [0, 5], [5, 6], [6, 7], [7, 8],
    # Middle (9-12)
    [0, 9], [9, 10], [10, 11], [11, 12],
    # Ring (13-16)
    [0, 13], [13, 14], [14, 15], [15, 16],
    # Pinky (17-20)
    [0, 17], [17, 18], [18, 19], [19, 20]
]

def load_model(path=''):
    net = HandJointsDetection(joint_map=JOINTS_MAP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.to(device)

    return net

def load_HanCo_ds(batch_size=48, num_samples_per_epoch=24000):

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
    
    train_data = HanCoDataset(root='HanCo_dataset',
                              transform=transform)

    sampler = RandomSampler(
        train_data, 
        replacement=False, 
        num_samples=num_samples_per_epoch
    )

    train_dl = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)

    val_data = FreiHand(root=os.path.join('FreiHand_eval_dataset', 'FreiHAND_pub_v2_eval'),
                                  transform=transform)
    
    val_dl = DataLoader(val_data, batch_size=2*batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_dl, val_dl