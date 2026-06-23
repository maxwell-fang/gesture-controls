from model import HandJointsDetection
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import HandsDataset
from tqdm import tqdm

def visualize_loss(train_losses, val_losses, file_name=''):
    plt.figure()
    if train_losses != []:
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    if val_losses != []:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', color='red')
    
    plt.title('Hand Keypoint Model - Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def visualize_loss_clean(train_losses, val_losses, file_name=''):
    plt.figure()
    if train_losses != []:
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    if val_losses != []:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', color='red')
    
    plt.title('Hand Keypoint Model - Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def train(batch_size, epochs):

    device = torch.device('cuda')

    image_path = '../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/images/train'
    annfile_path = '../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/coco_annotation/train/_annotations.coco.json'
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = HandsDataset(root=image_path, annFile=annfile_path, transform=transform, dataset_name='Hand Keypoints Training Set')
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

    val_path = '../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/images/val'
    val_annfile_path = '../hand_keypoints_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/coco_annotation/val/_annotations.coco.json'
    val_data = HandsDataset(root=val_path, annFile=val_annfile_path, transform=transform, dataset_name='Hand Keypoints Validation Set')
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

    net = HandJointsDetection(img_size=224, embedding_dim=3)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fcn = net.loss_fcn
    net.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = torch.inf

    for epoch in tqdm(range(epochs)):

        net.train()
        epoch_train_loss = 0.0
        for train_inputs, train_labels in tqdm(train_dl):
            heatmaps, visibility = train_labels
            target_heatmaps = heatmaps.to(device)
            visibility = visibility.to(device)
            train_samples = train_inputs.to(device)
            train_outputs = net(train_samples)

            train_loss = loss_fcn(train_outputs, target_heatmaps, visibility)
    
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item()

        train_losses.append(epoch_train_loss/len(train_dl))

        net.eval()
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
            torch.save(net.state_dict(), f'./models/{epoch + 1}.pt')

        val_losses.append(epoch_val_loss/len(val_dl))

        print(f"Epoch [{epoch+1}/{epochs}] Finished. Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        visualize_loss(train_losses=train_losses, val_losses=val_losses, file_name='./plots/loss_curve.png')
    torch.save(net.state_dict(), f'./models/{epoch}.pt')

if __name__ == '__main__':
    train(batch_size=48, epochs=100)