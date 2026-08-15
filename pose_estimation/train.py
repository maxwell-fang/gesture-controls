from pose_estimation import HandJointsDetection, JOINTS_MAP, _soft_argmax_2d, write_pose_predictions, load_HanCo_ds
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import gc

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


# def train_single_batch(batch_size: int, epochs: int, model: HandJointsDetection | None, start_epoch=0, learning_rate=0.0001, load_weights=False, patience=20):

#     device = torch.device('cuda')

#     train_dl, val_dl = load_merged_ds(batch_size=batch_size)

#     if not isinstance(model, HandJointsDetection):
#         net = HandJointsDetection(img_size=224, embedding_dim=3, joint_map=JOINTS_MAP)
#         train_losses = []
        
#     elif load_weights:
#         net = model
#         print('Model Loaded')
#         train_losses = []

#     else:
#         net = model
#         print('Model Loaded')
#         with open('./pose_estimation/train_losses.json', 'r') as f:
#             train_losses = json.load(f)

#     optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
#     loss_fcn = net.loss_fcn
#     net.to(device)

#     train_inputs, train_labels = next(iter(train_dl))

#     keypoints, heatmaps, visibility = train_labels
#     target_heatmaps = heatmaps.to(device)
#     visibility = visibility.to(device)
#     train_samples = train_inputs.to(device)
#     train_keypoints = keypoints.to(device)

#     val_inputs, val_labels = next(iter(val_dl))
#     for epoch in tqdm(range(start_epoch, epochs + start_epoch)):

#         net.train()
#         epoch_train_loss = 0.0
#         train_outputs = net(train_samples)

#         train_loss = loss_fcn(train_outputs, train_keypoints, target_heatmaps, visibility)

#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()

#         epoch_train_loss = train_loss.item()

#         train_losses.append(epoch_train_loss)
#         print(f"Heatmap Min: {train_outputs.min().item():.4f}, Max: {train_outputs.max().item():.4f}")

#         if (epoch + 1) % 10 == 0:
#             net.eval()
#             with torch.no_grad():
#                 train_outputs = net(train_samples)
#                 sample_heatmaps = train_outputs[0]
#                 pred_coords_norm = _soft_argmax_2d(sample_heatmaps.unsqueeze(0), temperature=0.1)[0]
#                 pred_coords_px = (pred_coords_norm * 224.0).cpu()
#                 fixed_vis_img = train_inputs[0]
#                 # Draw on the exact same sample 0 image saved earlier
#                 pred_vis = write_pose_predictions(fixed_vis_img, pred_coords_px, JOINTS_MAP)
#                 cv2.imwrite(f'./debug_overfit_epoch_{epoch+1}.png', pred_vis)

#         print(f"Epoch [{epoch+1}/{epochs+start_epoch}] Finished. Train Loss: {epoch_train_loss:.4f}")

# def train_merged(batch_size: int, epochs: int, model: HandJointsDetection | None, start_epoch=0, learning_rate=0.0001, load_weights=False, patience=20):

#     device = torch.device('cuda')

#     train_dl, val_dl = load_augmented_merged_ds(batch_size=batch_size)

#     if not isinstance(model, HandJointsDetection):
#         net = HandJointsDetection(img_size=224, embedding_dim=3, joint_map=JOINTS_MAP)
#         train_losses = []
#         val_losses = []
        
#     elif load_weights:
#         net = model
#         print('Model Loaded')
#         train_losses = []
#         val_losses = []

#     else:
#         net = model
#         print('Model Loaded')
#         with open('./pose_estimation/train_losses.json', 'r') as f:
#             train_losses = json.load(f)
#         with open('./pose_estimation/val_losses.json', 'r') as f:
#             val_losses = json.load(f)
        

#     optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.05)
#     loss_fcn = net.loss_fcn
#     net.to(device)
    
#     best_val_loss = torch.inf
#     patience_ind = 0
#     prev_val_loss = torch.inf

#     for epoch in tqdm(range(start_epoch, epochs + start_epoch)):

#         net.train()
#         epoch_train_loss = 0.0
#         for train_inputs, train_labels in tqdm(train_dl):
#             keypoints, heatmaps, visibility = train_labels
#             target_heatmaps = heatmaps.to(device)
#             visibility = visibility.to(device)
#             train_samples = train_inputs.to(device)
#             train_keypoints = keypoints.to(device)
#             train_outputs = net(train_samples)

#             train_loss = loss_fcn(train_outputs, train_keypoints, target_heatmaps, visibility)

#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()

#             epoch_train_loss += train_loss.item()

#         train_losses.append(epoch_train_loss/len(train_dl))

#         net.eval()
#         epoch_val_loss = 0.0
#         with torch.no_grad():
#             for val_inputs, val_labels in tqdm(val_dl):
#                 keypoints, heatmaps, visibility = val_labels
#                 val_target_heatmaps = heatmaps.to(device)
#                 val_visibility = visibility.to(device)
#                 val_samples = val_inputs.to(device)
#                 val_keypoints = keypoints.to(device)
#                 val_outputs = net(val_samples)

#                 val_loss = loss_fcn(val_outputs, val_keypoints, val_target_heatmaps, val_visibility)

#                 epoch_val_loss += val_loss.item()

#             torch.save(net.state_dict(), f'./pose_estimation/models/{epoch + 1}.pt')

#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             torch.save(net.state_dict(), f'./pose_estimation/models/best.pt')

#         if abs(epoch_val_loss - prev_val_loss) < 1e-3:
#             patience_ind += 1
#         else:
#             patience_ind = 0

#         val_losses.append(epoch_val_loss/len(val_dl))

#         print(f"Epoch [{epoch+1}/{epochs+start_epoch}] Finished. Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
#         with open('./pose_estimation/train_losses.json', 'w') as f:
#             json.dump(train_losses, f, indent=4)
#         with open('./pose_estimation/val_losses.json', 'w') as f:
#             json.dump(val_losses, f, indent=4)

#         visualize_loss(train_losses=train_losses, val_losses=val_losses, file_name='./pose_estimation/plots/loss_curve.png')

#         if patience_ind >= patience:
#             print(f'Model has not improved significantly for {patience} epochs.')
#             return

#         prev_val_loss = epoch_val_loss

#     torch.save(net.state_dict(), f'./pose_estimation/models/{epoch + 1}.pt')

def train_HanCo(batch_size: int, epochs: int, model: HandJointsDetection | None, start_epoch=0, learning_rate=0.0001, load_weights=False, patience=20):

    device = torch.device('cuda')

    train_dl, val_dl = load_HanCo_ds(batch_size=batch_size, num_samples_per_epoch=16000)

    if not isinstance(model, HandJointsDetection):
        net = HandJointsDetection(img_size=224, no_stacks=3, embedding_dim=3, joint_map=JOINTS_MAP)
        train_losses = []
        val_losses = []
        
    elif load_weights:
        net = model
        print('Model Loaded')
        train_losses = []
        val_losses = []

    else:
        net = model
        print('Model Loaded')
        with open('./pose_estimation/train_losses.json', 'r') as f:
            train_losses = json.load(f)
        with open('./pose_estimation/val_losses.json', 'r') as f:
            val_losses = json.load(f)
        

    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.05)
    loss_fcn = net.loss_fcn
    net.to(device)
    
    best_val_loss = torch.inf
    patience_ind = 0
    prev_val_loss = torch.inf

    for epoch in tqdm(range(start_epoch, epochs + start_epoch)):

        net.train()
        epoch_train_loss = 0.0
        for train_data in tqdm(train_dl):
            train_samples = train_data['image'].to(device)
            train_keypoints = train_data['keypoints_2d_norm'].to(device)
            visibility = train_data['visibility'].to(device)
            target_heatmaps = train_data['target_heatmaps'].to(device)
            train_heatmaps, train_outputs = net(train_samples)

            train_loss = loss_fcn(train_heatmaps, train_keypoints, target_heatmaps, visibility)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item()

        train_losses.append(epoch_train_loss/len(train_dl))

        del train_samples, train_keypoints, visibility, target_heatmaps, train_heatmaps, train_outputs, train_loss
        gc.collect()
        torch.cuda.empty_cache()

        net.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for val_data in tqdm(val_dl):
                val_samples = val_data['image'].to(device)
                val_keypoints = val_data['keypoints_2d_norm'].to(device)
                val_visibility = val_data['visibility'].to(device)
                val_target_heatmaps = val_data['target_heatmaps'].to(device)
                val_heatmaps = net(val_samples)
                val_loss = loss_fcn(val_heatmaps, val_keypoints, val_target_heatmaps, val_visibility)

                epoch_val_loss += val_loss.item()

            torch.save(net.state_dict(), f'./pose_estimation/models/{epoch + 1}.pt')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(net.state_dict(), f'./pose_estimation/models/best.pt')

        if abs(epoch_val_loss - prev_val_loss) < 1e-3:
            patience_ind += 1
        else:
            patience_ind = 0

        val_losses.append(epoch_val_loss/len(val_dl))

        print(f"Epoch [{epoch+1}/{epochs+start_epoch}] Finished. Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        with open('./pose_estimation/train_losses.json', 'w') as f:
            json.dump(train_losses, f, indent=4)
        with open('./pose_estimation/val_losses.json', 'w') as f:
            json.dump(val_losses, f, indent=4)

        visualize_loss(train_losses=train_losses, val_losses=val_losses, file_name='./pose_estimation/plots/loss_curve.png')

        if patience_ind >= patience:
            print(f'Model has not improved significantly for {patience} epochs.')
            return

        prev_val_loss = epoch_val_loss

        del val_samples, val_keypoints, val_visibility, val_target_heatmaps, val_heatmaps, val_loss
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(net.state_dict(), f'./pose_estimation/models/{epoch + 1}.pt')