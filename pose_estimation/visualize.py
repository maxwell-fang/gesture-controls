import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from pose_estimation import HandJointsDetection, generate_heatmaps, load_model

def visualize_pred(image:torch.Tensor, label:torch.Tensor, joint_map:list[list[int]], model: HandJointsDetection):

    channels, H, W = image.size()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))

    heatmaps, visibility = generate_heatmaps(imgsize=(H, W), keypoints=label, std=1, downscale_factor=4)
    keypoints = heatmaps_to_coords(heatmaps) * 4
    print(outputs.size())
    output = outputs[:, :]
    output_coords = heatmaps_to_coords(output) * 4
    processed_img = write_predictions_labels(image, keypoints, output_coords, joint_map, 1, 2, 4)

    return processed_img

def visualize_predictions(images:torch.Tensor, labels:torch.Tensor, joint_map:list[list[int]], model: str | HandJointsDetection):
    
    if isinstance(model, str):
        net = load_model(model)
        net.eval()
    else:
        net = model

    no_imgs, channels, H, W = images.size()
    processed_images = torch.zeros((no_imgs, channels, H, W))

    for ind in tqdm(range(no_imgs)):
        img = images[ind, :, :, :]
        label = labels[ind, :, :]

        processed_images[ind, :, :, :] = visualize_pred(img, label, joint_map, net)

    return processed_images

def write_predictions_labels(image, labels, outputs, joint_map, ptsize, lnsize):

    label = labels.detach().cpu().numpy().astype(np.uint8) if isinstance(labels, torch.Tensor) else np.array(labels, dtype=np.uint8)
    output = outputs.detach().cpu().numpy().astype(np.uint8) if isinstance(outputs, torch.Tensor) else np.array(outputs, dtype=np.uint8)

    if label.shape != (21, 2):
        label = label.reshape(21, 2)

    if outputs.shape != (21, 2):
        output = output.reshape(21, 2)

    
    if type(image) == torch.Tensor:
        img = image.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)
        img = img * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image

    no_points, __ = label.shape

    for ind in range(no_points):
        label_pt = label[ind, :][::-1]
        output_pt = output[ind, :][::-1]

        cv2.circle(img, label_pt, ptsize, (0, 255, 0), -1)
        cv2.circle(img, output_pt, ptsize, (0, 0, 255), -1)

    for joints in joint_map:
        label_jt_1 = label[joints[0], :][::-1].tolist()
        label_jt_2 = label[joints[1], :][::-1].tolist()

        output_jt_1 = output[joints[0], :][::-1].tolist()
        output_jt_2 = output[joints[1], :][::-1].tolist()

        img = cv2.line(img, label_jt_1, label_jt_2, (0, 255, 0), lnsize)
        img = cv2.line(img, output_jt_1, output_jt_2, (0, 0, 255), lnsize)

    return img

def write_pose_predictions(image, outputs, joint_map, ptsize=3, lnsize=2):
    # 1. Handle Torch Tensor -> BGR OpenCV Image conversion safely
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
        if img.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            img = img.transpose(1, 2, 0)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        # PyTorch images are RGB, convert to BGR for OpenCV drawing/saving
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    # 2. Extract coordinates while preserving float values
    if isinstance(outputs, torch.Tensor):
        coords = outputs.detach().cpu().numpy()
    else:
        coords = np.array(outputs)

    coords = coords.reshape(21, 2)

    # 3. Draw Bone Connections
    for joint_a, joint_b in joint_map:
        pt1 = (int(round(coords[joint_a, 0])), int(round(coords[joint_a, 1])))
        pt2 = (int(round(coords[joint_b, 0])), int(round(coords[joint_b, 1])))
        cv2.line(img, pt1, pt2, (0, 0, 255), lnsize)

    # 4. Draw Joint Points
    for ind in range(coords.shape[0]):
        pt = (int(round(coords[ind, 0])), int(round(coords[ind, 1])))
        cv2.circle(img, pt, ptsize, (0, 255, 0), -1)

    return img

def heatmaps_to_coords(heatmaps):

    no_maps, H, W = heatmaps.size()

    keypoints = torch.zeros(size=[no_maps, 2], dtype=heatmaps.dtype)

    for i in range(no_maps): 
        map = heatmaps[i, :, :]
        flat_idx = torch.argmax(map)
        coords = torch.unravel_index(flat_idx, map.shape)
        keypoints[i, 0] = coords[0].item()
        keypoints[i, 1] = coords[1].item()

    return keypoints

def visualize_preds_video(video_path: str, model_path: str, joint_map: list[list[int]], target_size=(224, 224), output_video_path='./predictions/output_prediction.mp4'):
    """
    Loads an MP4 video using OpenCV, scales down the resolution, converts it 
    into a PyTorch tensor, runs inference frame-by-frame, and saves the output as a video.
    """
    # 1. Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(model_path)
    net.eval()
    net.to(device)

    # 2. Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Extract original video specs
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H, W = target_size

    # 3. Initialize the Video Writer to save your output MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    print(f"Processing video: {video_path} ({total_frames} frames)...")
    
    # Define the transform to handle scaling and normalize image to [0, 1] tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    # 4. Loop through frames
    for ind in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break  # Break if video ends abruptly
        
        # OpenCV reads BGR. Convert to RGB for the model/torchvision pipeline
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Scale resolution down and convert to torch tensor [3, H, W]
        img_tensor = transform(frame_rgb).to(device)
        
        # Add a batch dimension -> [1, 3, H, W] as expected by your neural network
        img_batch = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = net(img_batch)  # Run model inference
            output = output[0, :, :] # Remove batch dim -> [21, H, W] or similar heatmap dims

        output_coords = heatmaps_to_coords(output) * 4

        # Note: Since this is an un-annotated video file, we don't have ground truth labels.
        # We dummy-fill a blank tensor for the labels argument to fit your write_predictions_labels function.
        dummy_label = torch.zeros((21, 2)) 

        # 5. Draw the keypoints using your existing function
        # (It automatically converts the tensor back to BGR cv2 image internally)
        processed_img = write_predictions_labels(
            image=img_tensor, 
            labels=dummy_label, 
            outputs=output_coords, 
            joint_map=joint_map, 
            ptsize=2,  # Boosted size slightly since frames might be tiny
            lnsize=2
        )

        # 6. Write the frame into the output video
        out.write(processed_img)

    # Clean up resources
    cap.release()
    out.release()
    print(f"Finished! Video saved to {output_video_path}")

def pred_to_original_size(original_sz, input_sz, preds):
    
    ratio = original_sz/input_sz
    corrected_preds = preds*ratio
    
    return corrected_preds