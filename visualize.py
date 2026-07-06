import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from hand_detection.utils import pad_bb_square
from pose_estimation.utils import pred_to_original_size
from pose_estimation.visualize import write_predictions
from pose_estimation.model import HandJointsDetection
import torchvision.transforms.functional as TF

def write_predictions(hand_det_model: YOLO, pose_est_model: HandJointsDetection, in_video_path: str, out_video_path: str):

    # Open your video file
    cap = cv2.VideoCapture(in_video_path)
    assert cap.isOpened(), "Error reading video file"

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run inference on the current frame
        results = hand_det_model.predict(frame, verbose=False)

        for r in results:
            annotator = Annotator(frame)
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates in (left, top, right, bottom) format
                b = box.xyxy[0]
                hand_image, coords = pad_bb_square(frame, b)
                input_image = TF.to_tensor(cv2.resize(hand_image, pose_est_model.img_size))
                joints = pose_est_model(input_image)
                corrected_joints = pred_to_original_size(coords[2]-coords[0], pose_est_model.img_size, joints)

                conf = float(box.conf[0])
                label = f"{hand_det_model.names[0]} {conf:.2f}"
                annotator.box_label(b, label)

            frame = annotator.result()
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
