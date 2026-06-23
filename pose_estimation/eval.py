from model import HandJointsDetection
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import HandsDataset
from tqdm import tqdm
from pathlib import Path
from train import visualize_loss
from utils import load_model


def evaluate(models_path='', val_path='', val_annfile_path='', batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    val_data = HandsDataset(root=val_path, annFile=val_annfile_path, transform=transform, dataset_name='Hand Keypoints Validation Set')
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

    folder = Path(models_path)
    best_val_loss = torch.inf
    for item in folder.iterdir():
        print(item.name)
        print(models_path + '/' + item.name)
        net = load_model(models_path + '/' + item.name)
        loss_fcn = net.loss_fcn

        epoch_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in tqdm(val_dl):
                heatmaps, visibility = val_labels
                val_target_heatmaps = heatmaps.to(device)
                val_visibility = visibility.to(device)
                val_samples = val_inputs.to(device)
                val_outputs = net(val_samples)
                
                val_loss = loss_fcn(val_outputs, val_target_heatmaps, val_visibility)

                epoch_val_loss += val_loss.item()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = item.name
            print(f'model {best_model} is currently the best model with val loss: {epoch_val_loss}.')

def evaluate_batch(models_path='', val_path='', val_annfile_path='', batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    val_data = HandsDataset(root=val_path, annFile=val_annfile_path, transform=transform, dataset_name='Hand Keypoints Validation Set')
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

    folder = Path(models_path)
    best_val_loss = torch.inf
    losses = []

    val_inputs, val_labels = next(iter(val_dl))

    for item in tqdm(folder.iterdir()):
        net = load_model(models_path + '/' + item.name)
        loss_fcn = net.loss_fcn

        epoch_val_loss = 0.0
        with torch.no_grad():
            heatmaps, visibility = val_labels
            val_target_heatmaps = heatmaps.to(device)
            val_visibility = visibility.to(device)
            val_samples = val_inputs.to(device)
            val_outputs = net(val_samples)
            
            val_loss = loss_fcn(val_outputs, val_target_heatmaps, val_visibility)

            epoch_val_loss += val_loss.item()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = item.name
            losses.append(epoch_val_loss)

    print(f'The best model is {best_model}.')
    visualize_loss(train_losses=[], val_losses=losses, file_name='./plots/batch_losses.png')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # evaluate(models_path='./models',
    #                val_path='../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/images/val',
    #                val_annfile_path='../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/coco_annotation/val/_annotations.coco.json',
    #                batch_size=16)

    # evaluate_batch(models_path='./models',
    #                val_path='../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/images/val',
    #                val_annfile_path='../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/coco_annotation/val/_annotations.coco.json',
    #                batch_size=16)
    val_path='../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/images/val'
    val_annfile_path='../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/coco_annotation/val/_annotations.coco.json'

    transform = transforms.Compose([transforms.ToTensor()])
    val_data = HandsDataset(root=val_path, annFile=val_annfile_path, transform=transform, dataset_name='Hand Keypoints Validation Set', heatmaps=False)
    val_dl = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

    val_inputs, val_labels = next(iter(val_dl))
    images = val_inputs.to(device)
    
    visualize_predictions(images, val_labels, './models/65.pt', val_data.joints_map)
    