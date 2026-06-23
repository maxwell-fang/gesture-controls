from model import HandJointsDetection
import torch

def load_model(path=''):
    net = HandJointsDetection()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.to(device)

    return net

def read_video(path=''):
    # video = decoders.VideoDecoder(source=path)
    pass