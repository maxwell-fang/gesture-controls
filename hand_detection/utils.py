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