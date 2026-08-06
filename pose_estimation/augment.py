import albumentations as A
import cv2
import json
import os
import torch
from collections import defaultdict

def augment_individual_jsons(img_dir, json_dir, output_img_dir, output_json_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    transform = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 5), 
        hole_height_range=(8, 32), 
        hole_width_range=(8, 32), 
        fill=0, 
        p=0.5
    )],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'):
            continue
            
        base_name = os.path.splitext(json_file)[0]
        
        img_name = None
        for ext in ['.jpg', '.jpeg', '.png']:
            if os.path.exists(os.path.join(img_dir, base_name + ext)):
                img_name = base_name + ext
                break
                
        if not img_name:
            continue
            
        image = cv2.imread(os.path.join(img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        with open(os.path.join(json_dir, json_file), 'r') as f:
            label_data = json.load(f)
            
        raw_pts = label_data['hand_pts']
        is_left = label_data['is_left']
        
        xy_pts = [[pt[0], pt[1]] for pt in raw_pts]
        vis_flags = [pt[2] for pt in raw_pts]
        
        try:
            transformed = transform(image=image, keypoints=xy_pts)
            aug_image = transformed['image']
            aug_xy_pts = transformed['keypoints']
            
            final_hand_pts = []
            for i, (aug_x, aug_y) in enumerate(aug_xy_pts):
                if aug_x < 0 or aug_x >= w or aug_y < 0 or aug_y >= h:
                    current_vis = 0.0
                else:
                    current_vis = float(vis_flags[i])
                    
                final_hand_pts.append([round(aug_x, 4), round(aug_y, 4), current_vis])
                
            new_label_data = {
                "hand_pts": final_hand_pts,
                "is_left": is_left
            }
            
            aug_name = f"aug_{base_name}"
            cv2.imwrite(
                os.path.join(output_img_dir, f"{aug_name}.jpg"), 
                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            )
            
            with open(os.path.join(output_json_dir, f"{aug_name}.json"), 'w') as f:
                json.dump(new_label_data, f, indent=4)
                
        except Exception as e:
            print(f"Skipping file {json_file} due to processing error: {e}")

def augment_coco_keypoints(json_path, img_dir, output_img_dir, output_json_path):
    os.makedirs(output_img_dir, exist_ok=True)

    transform_coco = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 5), 
        hole_height_range=(8, 32), 
        hole_width_range=(8, 32), 
        fill=0, 
        p=0.5
    )], 
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    bbox_params=A.BboxParams(format='coco', label_fields=['bbox_labels'], min_visibility=0.3, clip=True))
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        
    new_annotations = []
    ann_id_counter = 1

    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        anns = img_to_anns[img_id]
        if not anns:
            continue
            
        for ann in anns:
            raw_coco_kpts = ann.get('keypoints', [])
            if not raw_coco_kpts:
                continue
                
            xy_pts = []
            vis_flags = []
            for i in range(0, len(raw_coco_kpts), 3):
                xy_pts.append([raw_coco_kpts[i], raw_coco_kpts[i+1]])
                vis_flags.append(raw_coco_kpts[i+2])
                
            # Extract optional bounding box
            bboxes = []
            bbox_labels = []
            if 'bbox' in ann:
                x, y, w_box, h_box = ann['bbox']
                
                # NOTE: If your JSON boxes are normalized (0.0 to 1.0), you MUST scale them 
                # up to pixel coordinates here so 'coco' format reads them correctly:
                if max(ann['bbox']) <= 1.01:
                    x *= w
                    y *= h
                    w_box *= w
                    h_box *= h

                # Force clamp to stay safely within physical image pixel boundaries
                x = max(0.0, min(float(x), float(w - 1)))
                y = max(0.0, min(float(y), float(h - 1)))
                w_box = max(1.0, min(float(w_box), float(w - x)))
                h_box = max(1.0, min(float(h_box), float(h - y)))
                
                if x + w_box >= w:
                    w_box = float(w - x - 0.1)
                if y + h_box >= h:
                    h_box = float(h - y - 0.1)

                bboxes = [[x, y, w_box, h_box]]
                bbox_labels = [ann.get('category_id', 1)]
            
            try:
                transformed = transform_coco(
                    image=image, 
                    keypoints=xy_pts, 
                    bboxes=bboxes, 
                    bbox_labels=bbox_labels
                )
                aug_image = transformed['image']
                aug_xy_pts = transformed['keypoints']
                aug_bboxes = transformed['bboxes']
                
                final_coco_kpts = []
                for idx, (aug_x, aug_y) in enumerate(aug_xy_pts):
                    current_vis = vis_flags[idx]
                    
                    if aug_x < 0 or aug_x >= w or aug_y < 0 or aug_y >= h:
                        current_vis = 0
                        aug_x, aug_y = 0.0, 0.0
                        
                    final_coco_kpts.extend([round(aug_x, 2), round(aug_y, 2), current_vis])
                
                aug_img_name = f"aug_{img_name}"
                cv2.imwrite(
                    os.path.join(output_img_dir, aug_img_name), 
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )
                
                new_ann = ann.copy()
                new_ann['id'] = ann_id_counter
                new_ann['keypoints'] = final_coco_kpts
                
                if aug_bboxes:
                    new_ann['bbox'] = [round(x, 2) for x in aug_bboxes[0]]
                    new_ann['area'] = new_ann['bbox'][2] * new_ann['bbox'][3]
                
                new_annotations.append(new_ann)
                ann_id_counter += 1
                
                img_info['file_name'] = aug_img_name
                
            except Exception as e:
                print(f"Skipping COCO image {img_name} due to an error: {e}")

    coco_data['annotations'] = new_annotations
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)