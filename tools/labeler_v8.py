import os
import json
from ultralytics import YOLO
from tqdm import tqdm
import torch

# Configuration
MODEL_PATH = 'model.pt'
SOURCE_DIR = 'dataset/800'
VIZ_DIR = os.path.abspath('dataset/labeled_viz')
OUTPUT_JSON = 'dataset/labels_auto.json'

# Mapping dictionary for names
CLASS_MAP = {0: 'Darts', 1: 'Corner1', 2: 'Corner2', 3: 'Corner3', 4: 'Corner4'}

def generate():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found!")
        return

    print("🚀 Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    
    final_labels = {}
    idx = 0
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: {SOURCE_DIR} not found!")
        return
        
    folders = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))])
    
    print(f"📁 Found {len(folders)} folders to process.")

    for folder in tqdm(folders, desc="Processing Dataset"):
        source_folder = os.path.join(SOURCE_DIR, folder)
        images = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not images: continue
        
        # Disable project-wide saving, we will do it manually for custom look
        results = model.predict(source_folder, save=False, verbose=False, device='cpu')
        
        folder_viz = os.path.join(VIZ_DIR, folder)
        if not os.path.exists(folder_viz): os.makedirs(folder_viz)

        for result in results:
            img_name = os.path.basename(result.path)
            orig_img = result.orig_img.copy() # Get original image
            h, w = orig_img.shape[:2]
            
            points = [None] * 4 
            dart_points = []     
            
            # Colors matching Rust: C1=Red, C2=Green, C3=Blue, C4=Yellow, D=White
            # OpenCV uses BGR: (B, G, R)
            COLORS = [
                (0, 0, 255),    # C1: Red
                (0, 255, 0),    # C2: Green
                (255, 0, 0),    # C3: Blue
                (0, 255, 255),  # C4: Yellow
            ]
            DARTS_COLOR = (255, 255, 255) # White

            if result.boxes is not None:
                boxes = result.boxes.xyxyn.cpu().numpy() 
                classes = result.boxes.cls.cpu().numpy()
                
                import cv2
                for box, cls in zip(boxes, classes):
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    p = [float(cx), float(cy)]
                    
                    c_idx = int(cls)
                    px, py = int(cx * w), int(cy * h)

                    # Draw custom 12x12 box and label
                    color = DARTS_COLOR
                    label_text = "D"
                    
                    if 1 <= c_idx <= 4:
                        points[c_idx - 1] = p
                        color = COLORS[c_idx - 1]
                        label_text = f"C{c_idx}"
                    elif c_idx == 0:
                        dart_points.append(p)

                    # Draw Box (12x12)
                    cv2.rectangle(orig_img, (px-6, py-6), (px+6, py+6), color, 2)
                    # Draw Label
                    cv2.putText(orig_img, label_text, (px-6, py-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Save manual viz
            cv2.imwrite(os.path.join(folder_viz, img_name), orig_img)

            # Build final coordinate list for JSON
            final_xy = []
            for p in points:
                if p is None: final_xy.append([0.0, 0.0])
                else: final_xy.append(p)
            final_xy.extend(dart_points)

            final_labels[str(idx)] = {
                "img_folder": folder,
                "img_name": img_name,
                "bbox": [0, 800, 0, 800],
                "xy": final_xy
            }
            idx += 1

    # Save final JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_labels, f, indent=2)
    
    print(f"\n✅ All Done! Labeled images saved at {VIZ_DIR}")
    print(f"📜 JSON Labels saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    generate()
