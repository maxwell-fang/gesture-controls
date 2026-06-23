from ultralytics import YOLO

def train():

    model = YOLO("yolo26n.pt")

    results = model.train(data='', epochs=200, imgsz=0, patience=5)