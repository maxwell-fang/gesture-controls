import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from hand_detection import pad_bb_square, expand_bbox_for_wrist
from pose_estimation import write_pose_predictions, pred_to_original_size, HandJointsDetection, load_model, JOINTS_MAP, heatmaps_to_coords
import torchvision.transforms.functional as TF
import numpy as np

def write_predictions_video(hand_det_model: YOLO, pose_est_model: HandJointsDetection, joints_map: list[list[int]], in_video_path: str, out_video_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Open your video file
    cap = cv2.VideoCapture(in_video_path)
    assert cap.isOpened(), "Error reading video file"

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    pose_in_sz = (pose_est_model.img_size, pose_est_model.img_size)
    # Initialize video writer
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run inference on the current frame
        results = hand_det_model.predict(frame, verbose=False)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for r in results:
            # annotator = Annotator(frame)
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates in (left, top, right, bottom) format
                b = box.xyxy[0]
                center_x, center_y, __, __ = box.xywh[0]
                b = expand_bbox_for_wrist(b, frame.shape, 0.40)
                hand_image, coords = pad_bb_square(frame, b)
                input_image = TF.to_tensor(cv2.resize(hand_image, pose_in_sz))
                input_image = input_image.to(device)

                with torch.no_grad():
                    joints = pose_est_model.predict(input_image)
                corrected_joints = joints*(coords[2]-coords[0])
                shift = torch.Tensor([[coords[0], coords[1]]])
                corrected_joints = corrected_joints + shift
                frame = write_pose_predictions(frame, corrected_joints, joints_map, 4, 5)

                frame = cv2.rectangle(frame, coords[:2], coords[2:4], 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    cap.release()
    out.release()

def write_predictions(hand_det_model: YOLO, pose_est_model: HandJointsDetection, joints_map: list[list[int]], in_image_path: str, out_image_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = cv2.imread(in_image_path)

    pose_in_sz = (pose_est_model.img_size, pose_est_model.img_size)

    results = hand_det_model.predict(frame, verbose=False)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates in (left, top, right, bottom) format
            b = box.xyxy[0]
            center_x, center_y, __, __ = box.xywh[0]
            hand_image, coords = pad_bb_square(frame, b)
            input_image = TF.to_tensor(cv2.resize(hand_image, pose_in_sz))
            print(input_image.size())
            input_image = input_image.to(device).unsqueeze(0)
            with torch.no_grad():
                joints = pose_est_model(input_image)
                joints = joints[0, :, :, :]
            input_image = input_image.squeeze(0)
            joints = heatmaps_to_coords(joints)
            print(joints)
            pose_image = write_pose_predictions(input_image, joints, joints_map, 4, 5)
            cv2.imwrite('./test2.png', pose_image)
            corrected_joints = pred_to_original_size(coords[2]-coords[0], pose_est_model.img_size, joints)
            shift = torch.Tensor([[center_y, center_x]])
            corrected_joints = corrected_joints + shift
            frame = write_pose_predictions(frame, corrected_joints, joints_map, 4, 5)
            conf = float(box.conf[0])
            label = f"{hand_det_model.names[0]} {conf:.2f}"
            # annotator.box_label(b, label)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = annotator.result()
    cv2.imwrite(out_image_path, frame)

   

if __name__ == '__main__':

    pose_est = load_model('./pose_estimation/models/best.pt')
    hand_detection = YOLO('./hand_detection/runs/detect/train/weights/best.pt')
    in_video_path = './hand_detection/output_video.mp4'
    in_image_path = './pose_estimation/test_samples/WIN_20260710_20_03_54_Pro.jpg'
    out_video_path = './demo_video.mp4'
    out_image_path = './pose_test.png' 
    write_predictions(hand_det_model=hand_detection,
                       pose_est_model=pose_est,
                       joints_map=JOINTS_MAP,
                       in_image_path=in_image_path,
                       out_image_path=out_image_path)