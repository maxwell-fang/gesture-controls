import torch

def pad_bb_square(frame, bb_coords):

    frame_h, frame_w, _ = frame.shape

    x1 = int(bb_coords[0].item())
    y1 = int(bb_coords[1].item())
    x2 = int(bb_coords[2].item())
    y2 = int(bb_coords[3].item())
    
    w = x2 - x1
    h = y2 - y1

    if w > h:
        pad_dist = (w - h) // 2
        y1 -= pad_dist
        y2 += pad_dist
    elif h > w:
        pad_dist = (h - w) // 2
        x1 -= pad_dist
        x2 += pad_dist

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0

    if x2 > frame_w:
        x1 -= (x2 - frame_w)
        x2 = frame_w
    if y2 > frame_h:
        y1 -= (y2 - frame_h)
        y2 = frame_h

    x1, y1 = max(0, x1), max(0, y1)

    new_image = frame[y1:y2, x1:x2, :]
    
    new_coords = [x1, y1, x2, y2]

    return new_image, new_coords


def expand_bbox_for_wrist(bbox, img_shape, extension_factor=0.40):
    """
    Expands the bottom/wrist area of the crop more aggressively 
    to capture the forearm context.
    """
    img_h, img_w = img_shape[:2]
    xmin, ymin, xmax, ymax = bbox
    
    h = ymax - ymin
    
    # 1. Expand width slightly to keep aspect ratio stable
    pad_w = (xmax - xmin) * 0.15
    new_xmin = max(0, int(xmin - pad_w))
    new_xmax = min(img_w, int(xmax + pad_w))
    
    # 2. Expand the top slightly (10%)
    new_ymin = max(0, int(ymin - (h * 0.10)))
    
    # 3. Push the BOTTOM down aggressively (40%) to grab the wrist/forearm
    new_ymax = min(img_h, int(ymax + (h * extension_factor)))
    
    return torch.Tensor([new_xmin, new_ymin, new_xmax, new_ymax])