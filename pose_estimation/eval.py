from pose_estimation import visualize_loss, load_model, JOINTS_MAP, expand_bbox_for_wrist, hard_argmax_2d, write_pose_predictions
from hand_detection import pad_bb_square
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import gc
import os
from ultralytics import YOLO
import cv2

# def evaluate(models_path='', val_path='', val_annfile_path='', batch_size=32):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([transforms.ToTensor()])
#     val_data = HandsDataset(root=val_path, annFile=val_annfile_path, transform=transform, dataset_name='Hand Keypoints Validation Set')
#     val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

#     folder = Path(models_path)
#     best_val_loss = torch.inf
#     for item in folder.iterdir():
#         print(item.name)
#         print(models_path + '/' + item.name)
#         net = load_model(models_path + '/' + item.name)
#         loss_fcn = net.loss_fcn

#         epoch_val_loss = 0.0
#         with torch.no_grad():
#             for val_inputs, val_labels in tqdm(val_dl):
#                 heatmaps, visibility = val_labels
#                 val_target_heatmaps = heatmaps.to(device)
#                 val_visibility = visibility.to(device)
#                 val_samples = val_inputs.to(device)
#                 val_outputs = net(val_samples)
                
#                 val_loss = loss_fcn(val_outputs, val_target_heatmaps, val_visibility)

#                 epoch_val_loss += val_loss.item()

#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             best_model = item.name
#             print(f'model {best_model} is currently the best model with val loss: {epoch_val_loss}.')

# def evaluate_single(models_path='', val_path='', val_annfile_path='', batch_size=32):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([transforms.ToTensor()])
#     val_data = HandsDataset(root=val_path, annFile=val_annfile_path, transform=transform, dataset_name='Hand Keypoints Validation Set')
#     val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)

#     folder = Path(models_path)
#     best_val_loss = torch.inf
#     losses = []

#     val_inputs, val_labels = next(iter(val_dl))

#     for item in tqdm(folder.iterdir()):
#         net = load_model(models_path + '/' + item.name)
#         net.eval()
#         loss_fcn = net.loss_fcn

#         epoch_val_loss = 0.0
#         with torch.no_grad():
#             heatmaps, visibility = val_labels
#             val_target_heatmaps = heatmaps.to(device)
#             val_visibility = visibility.to(device)
#             val_samples = val_inputs.to(device)
#             val_outputs = net(val_samples)
            
#             val_loss = loss_fcn(val_outputs, val_target_heatmaps, val_visibility)

#             epoch_val_loss += val_loss.item()

#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             best_model = item.name
        
#         losses.append(epoch_val_loss)

#         del net
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()

#     print(f'The best model is {best_model}.')
#     visualize_loss(train_losses=[], val_losses=losses, file_name='./pose_estimation/plots/batch_losses.png')

# def evaluate_batch(models_path='', val_path='', val_annfile_path='', batch_size=32, total_models=0):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([transforms.ToTensor()])
#     val_data = HandsDataset(root=val_path, annFile=val_annfile_path, transform=transform, dataset_name='Hand Keypoints Validation Set')
#     val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

#     folder = Path(models_path)
#     best_val_loss = torch.inf
#     losses = []

#     for item in tqdm(folder.iterdir(), total=total_models):
#         net = load_model(models_path + '/' + item.name)
#         loss_fcn = net.loss_fcn

#         epoch_val_loss = 0.0
#         for val_inputs, val_labels in val_dl:
#             with torch.no_grad():
#                 heatmaps, visibility = val_labels
#                 val_target_heatmaps = heatmaps.to(device)
#                 val_visibility = visibility.to(device)
#                 val_samples = val_inputs.to(device)
#                 val_outputs = net(val_samples)

#                 val_loss = loss_fcn(val_outputs, val_target_heatmaps, val_visibility)

#                 epoch_val_loss += val_loss.item()
#         avg_epoch_loss = epoch_val_loss / len(val_dl)
#         losses.append(avg_epoch_loss)

#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             best_model = item.name

#         del net
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()

#     print(f'The best model is {best_model}.')
#     visualize_loss(train_losses=[], val_losses=losses, file_name='./pose_estimation/plots/batch_losses.png')

def eval_all_models(models_path, image, output_path):

    models_lst = sorted(os.listdir(models_path))
    hand_det_model = YOLO('./hand_detection/runs/detect/train/weights/best.pt')
    joints_map=JOINTS_MAP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in models_lst:
        model_path = os.path.join(models_path, model)
        model_no = os.path.splitext(model)[0]
        net = load_model(model_path)

        pose_in_sz = (net.img_size, net.img_size)
        results = hand_det_model.predict(image, verbose=False)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for r in results:
            boxes = r.boxes
            for box in boxes:

                b = box.xyxy[0]
                center_x, center_y, __, __ = box.xywh[0]
                b = expand_bbox_for_wrist(b, image.shape, 0.40)
                hand_image, coords = pad_bb_square(image, b)

                input_image = transforms.functional.to_tensor(cv2.resize(hand_image, pose_in_sz))
                normalization = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                normalized_input_image = normalization(input_image)
                normalized_input_image = normalized_input_image.to(device).unsqueeze(0)

                with torch.no_grad():
                    heatmaps = net(normalized_input_image)

                B, C, H, W = heatmaps.shape
                input_image = input_image.squeeze(0)
                joints = hard_argmax_2d(heatmaps)
                joints = joints[0]*224
                joints = joints.to('cpu')
                input_image = (input_image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
                pose_image = write_pose_predictions(input_image, joints, joints_map, 1, 1)

                image_path = os.path.join(output_path, f'{model_no}.png')
                cv2.imwrite(image_path, pose_image)
    