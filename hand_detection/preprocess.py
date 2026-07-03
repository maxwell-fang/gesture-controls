import os
import glob
import cv2
import numpy as np
import scipy.io as sio

def convert_oxford_mat_to_yolo(mat_dir, images_dir, output_labels_dir):
    """
    Converts official Oxford Hand Dataset .mat annotations to YOLO txt format.
    Uses simplify_cells=True to eliminate NumPy indexing issues.
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {mat_dir}.")
        return

    for mat_path in mat_files:
        base_name = os.path.splitext(os.path.basename(mat_path))[0]
        img_path = os.path.join(images_dir, f"{base_name}.jpg")
        
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{base_name}.JPG")
            if not os.path.exists(img_path):
                continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w, _ = img.shape
        
        try:
            # simplify_cells converts the mat struct directly into clean Python lists/dicts
            mat_data = sio.loadmat(mat_path, simplify_cells=True)
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            continue
            
        if 'boxes' not in mat_data or mat_data['boxes'] is None:
            continue
            
        boxes = mat_data['boxes']
        
        # If there is only 1 hand instance, simplify_cells treats it as a single dict.
        # If there are multiple hands, it treats it as a list of dicts. We handle both:
        if isinstance(boxes, dict):
            boxes_list = [boxes]
        elif isinstance(boxes, list):
            boxes_list = boxes
        else:
            continue
            
        yolo_lines = []
        
        for hand in boxes_list:
            try:
                # With simplify_cells, hand['a'] is directly a standard list/array like [y, x] or [x, y]
                pt_a = hand['a']
                pt_b = hand['b']
                pt_c = hand['c']
                pt_d = hand['d']
                
                # --- DIAGONAL REFLECTION ASSIGNMENT ---
                # Index [1] forces X assignment and Index [0] forces Y assignment 
                # to correct the diagonal reflection issue.
                x_coords = [pt_a[1], pt_b[1], pt_c[1], pt_d[1]]
                y_coords = [pt_a[0], pt_b[0], pt_c[0], pt_d[0]]
                
                xmin, xmax = float(min(x_coords)), float(max(x_coords))
                ymin, ymax = float(min(y_coords)), float(max(y_coords))
                
                # Compute bounding properties
                w = xmax - xmin
                h = ymax - ymin
                x_center = xmin + (w / 2.0)
                y_center = ymin + (h / 2.0)
                
                # Canvas normalizations (YOLO relative format)
                x_center_norm = max(0.0, min(1.0, x_center / img_w))
                y_center_norm = max(0.0, min(1.0, y_center / img_h))
                w_norm = max(0.0, min(1.0, w / img_w))
                h_norm = max(0.0, min(1.0, h / img_h))
                
                if w_norm > 0.001 and h_norm > 0.001:
                    yolo_lines.append(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                
            except Exception:
                continue

        if yolo_lines:
            txt_output_path = os.path.join(output_labels_dir, f"{base_name}.txt")
            with open(txt_output_path, 'w') as f:
                f.write('\n'.join(yolo_lines))


def debug_visualize_labels(images_dir, labels_dir, num_samples=5):
    """
    Renders the newly generated YOLO txt annotations back onto your images 
    so you can visually confirm if the bounding boxes perfectly fit the hands.
    """
    print("\n--- Running Bounding Box Verification Check ---")
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))[:num_samples]
    
    for txt_path in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        img_path = os.path.join(images_dir, f"{base_name}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            _, xc, yc, wn, hn = map(float, parts)
            
            # Convert back from normalized YOLO format to pixels
            abs_w = int(wn * w)
            abs_h = int(hn * h)
            abs_x1 = int((xc * w) - (abs_w / 2))
            abs_y1 = int((yc * h) - (abs_h / 2))
            abs_x2 = abs_x1 + abs_w
            abs_y2 = abs_y1 + abs_h
            
            # Draw a green bounding box on the verification image
            cv2.rectangle(img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
            
        # Display the image window
        cv2.imshow("Verify Hand Label Orientation", img)
        print(f"Showing visual preview for {base_name}. Press ANY key to see the next one...")
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()

# --- DIRECTORY SETUP ---
# The Oxford dataset usually unzips into folders like 'training_dataset', 'validation_dataset', etc.
# Point these paths directly to one of those splits at a time:
MAT_ANNOTATIONS_DIR = "../hand_dataset/test_dataset/test_dataset/test_data/annotations"
IMAGES_DIR = "../hand_dataset/test_dataset/test_dataset/test_data/images"
OUTPUT_YOLO_LABELS_DIR = "../hand_dataset/test_dataset/labels"

if __name__ == "__main__":
    # convert_oxford_mat_to_yolo(MAT_ANNOTATIONS_DIR, IMAGES_DIR, OUTPUT_YOLO_LABELS_DIR)

    debug_visualize_labels('../hand_dataset/data/images/train', '../hand_dataset/data/labels/train', num_samples=3)