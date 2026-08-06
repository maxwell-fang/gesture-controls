from pose_estimation import HandJointsDetection, HanCoDataset, FreiHand
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
import os
import cv2

JOINTS_MAP = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
               [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17],
                 [17, 18], [18, 19], [19, 20]]

# JOINTS_MAP = [
#     # Thumb (1-4)
#     [0, 1], [1, 2], [2, 3], [3, 4],
#     # Index (5-8)
#     [0, 5], [5, 6], [6, 7], [7, 8],
#     # Middle (9-12)
#     [0, 9], [9, 10], [10, 11], [11, 12],
#     # Ring (13-16)
#     [0, 13], [13, 14], [14, 15], [15, 16],
#     # Pinky (17-20)
#     [0, 17], [17, 18], [18, 19], [19, 20]
# ]

def load_model(path=''):
    net = HandJointsDetection(joint_map=JOINTS_MAP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.to(device)

    return net

def load_augmented_merged_ds(batch_size=48):
    train_datasets = []
    val_datasets = []

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = HandsDataset(root='./merged_dataset/hand_keypoints/train/images',
                              annFile='./merged_dataset/hand_keypoints/train/annotations.json',
                              transform=transform,
                              dataset_name='Hand Keypoints Train',
                              heatmaps=True)
    train_datasets.append(train_data)

    train_data = HandsDataset(root='./merged_dataset/hand_keypoints_augmented/train/images',
                              annFile='./merged_dataset/hand_keypoints_augmented/train/annotations.json',
                              transform=transform,
                              dataset_name='Hand Keypoints Train',
                              heatmaps=True)
    train_datasets.append(train_data)

    cmu_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224), antialias=True)])
    train_data = CMUSyntheticHandDataset(img_dir='./merged_dataset/CMUSyntheticHands/train/images',
                                        json_dir='./merged_dataset/CMUSyntheticHands/train/labels',
                                        transform=cmu_transform,
                                        heatmaps=True)
    train_datasets.append(train_data)
    
    train_data = CMUSyntheticHandDataset(img_dir='./merged_dataset/CMUSyntheticHands_aug/train/images',
                                        json_dir='./merged_dataset/CMUSyntheticHands_aug/train/labels',
                                        transform=cmu_transform,
                                        heatmaps=True)
    train_datasets.append(train_data)

    merged_train = ConcatDataset(train_datasets)
    train_dl = DataLoader(merged_train, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

    val_data = HandsDataset(root='./merged_dataset/hand_keypoints/val/images',
                              annFile='./merged_dataset/hand_keypoints/val/annotations.json',
                              transform=transform,
                              dataset_name='Hand Keypoints Val',
                              heatmaps=True)
    val_datasets.append(val_data)

    val_data = HandsDataset(root='./merged_dataset/hand_keypoints_augmented/val/images',
                              annFile='./merged_dataset/hand_keypoints_augmented/val/annotations.json',
                              transform=transform,
                              dataset_name='Hand Keypoints Val',
                              heatmaps=True)
    val_datasets.append(val_data)

    val_data = CMUSyntheticHandDataset(img_dir='./merged_dataset/CMUSyntheticHands/val/images',
                                        json_dir='./merged_dataset/CMUSyntheticHands/val/labels',
                                        transform=cmu_transform,
                                        heatmaps=True)
    val_datasets.append(val_data)
    
    val_data = CMUSyntheticHandDataset(img_dir='./merged_dataset/CMUSyntheticHands_aug/val/images',
                                        json_dir='./merged_dataset/CMUSyntheticHands_aug/val/labels',
                                        transform=cmu_transform,
                                        heatmaps=True)
    val_datasets.append(val_data)
    
    merged_val = ConcatDataset(val_datasets)
    val_dl = DataLoader(merged_val, batch_size=2*batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_dl, val_dl


def load_merged_ds(batch_size=48):
    train_datasets = []
    val_datasets = []

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = HandsDataset(root='./merged_dataset/hand_keypoints/train/images',
                              annFile='./merged_dataset/hand_keypoints/train/annotations.json',
                              transform=transform,
                              dataset_name='Hand Keypoints Train',
                              heatmaps=True)
    train_datasets.append(train_data)

    cmu_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224), antialias=True)])
    train_data = CMUSyntheticHandDataset(img_dir='./merged_dataset/CMUSyntheticHands/train/images',
                                        json_dir='./merged_dataset/CMUSyntheticHands/train/labels',
                                        transform=cmu_transform,
                                        heatmaps=True)
    train_datasets.append(train_data)
    
    merged_train = ConcatDataset(train_datasets)
    train_dl = DataLoader(merged_train, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

    val_data = HandsDataset(root='./merged_dataset/hand_keypoints/val/images',
                              annFile='./merged_dataset/hand_keypoints/val/annotations.json',
                              transform=transform,
                              dataset_name='Hand Keypoints Val',
                              heatmaps=True)
    val_datasets.append(val_data)

    val_data = CMUSyntheticHandDataset(img_dir='./merged_dataset/CMUSyntheticHands/val/images',
                                        json_dir='./merged_dataset/CMUSyntheticHands/val/labels',
                                        transform=cmu_transform,
                                        heatmaps=True)
    val_datasets.append(val_data)
    
    merged_val = ConcatDataset(val_datasets)
    val_dl = DataLoader(merged_val, batch_size=2*batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_dl, val_dl

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

def expand_bbox_for_wrist(bbox, img_shape, extension_factor=0.40):
    """
    Expands the bottom/wrist area of the crop more aggressively 
    to capture the forearm context.
    """
    img_h, img_w = img_shape[:2]
    xmin, ymin, xmax, ymax = bbox
    
    h = ymax - ymin
    
    # 1. Expand width slightly to keep aspect ratio stable
    pad_w = (xmax - xmin) * 0.15
    new_xmin = max(0, int(xmin - pad_w))
    new_xmax = min(img_w, int(xmax + pad_w))
    
    # 2. Expand the top slightly (10%)
    new_ymin = max(0, int(ymin - (h * 0.10)))
    
    # 3. Push the BOTTOM down aggressively (40%) to grab the wrist/forearm
    new_ymax = min(img_h, int(ymax + (h * extension_factor)))
    
    return torch.Tensor([new_xmin, new_ymin, new_xmax, new_ymax])

def pad_to_simulate_distance(image, pad_percent=0.3):
    h, w, _ = image.shape
    pad_h = int(h * pad_percent)
    pad_w = int(w * pad_percent)
    
    # Add borders around the close-up image
    padded_img = cv2.copyMakeBorder(
        image, pad_h, pad_h, pad_w, pad_w, 
        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255] # Black background
    )
    return padded_img