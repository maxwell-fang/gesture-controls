from ultralytics import YOLO

def train(path, epochs=200):

    model = YOLO("yolo26n.pt")

    results = model.train(data=path, epochs=epochs, imgsz=500, patience=15)

if __name__ == '__main__':
    train('D:/projects/gesture-controls/hand_dataset/data.yaml', 1)
