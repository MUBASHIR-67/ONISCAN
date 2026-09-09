import cv2
from ultralytics import YOLO

IMAGE_PATH = "dataset/raw/Onion-Bad.jpg"
MODEL_PATH = "models/oni_scan_bad_onion/weights/best.pt"
OUTPUT_PATH = "dataset/processed/detected_onion.jpg"

model = YOLO(MODEL_PATH)

image = cv2.imread(IMAGE_PATH)

results = model(image, conf=0.25, verbose=False)
result = results[0]

if len(result.boxes) == 0:
    print("No onion detected.")
else:
    box = result.boxes.xyxy[0].cpu().numpy()

    x1, y1, x2, y2 = box.astype(int)

    crop = image[y1:y2, x1:x2]

    cv2.imwrite(OUTPUT_PATH, crop)

    print("Onion detected and cropped!")
    print("Box:", [x1, y1, x2, y2])
    print("Saved to:", OUTPUT_PATH)
    