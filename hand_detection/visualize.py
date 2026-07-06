import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load your model (use your custom .pt file here)
model = YOLO('./runs/detect/train/weights/best.pt')

# Open your video file
cap = cv2.VideoCapture('./test_samples/WIN_20260610_16_05_47_Pro.mp4')
assert cap.isOpened(), "Error reading video file"

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference on the current frame
    results = model.predict(frame, verbose=False)

    for r in results:
        annotator = Annotator(frame)
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates in (left, top, right, bottom) format
            b = box.xyxy[0]
            c = box.cls
            conf = float(box.conf[0])
            label = f"{model.names[int(c)]} {conf:.2f}"
            annotator.box_label(b, label)

        frame = annotator.result()
        out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
