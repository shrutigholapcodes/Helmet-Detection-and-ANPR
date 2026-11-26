
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# Load general YOLO model
model = YOLO("yolo11n.pt")

image_path = BASE_DIR / "uploads" / "test.jpeg"

# Detect motorcycles only
results = model.predict(
    source=str(image_path),
    classes=[3],
    conf=0.30,
    iou=0.40,
    imgsz=960,
    verbose=False
)

print("Motorcycles detected:", len(results[0].boxes))
for box in results[0].boxes:
    print(
        "Confidence:",
        round(float(box.conf[0]), 3),
        "| Bounding box:",
        box.xyxy[0].tolist()
    )


# Draw numbered motorcycle detections
annotated = results[0].orig_img.copy()

for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Motorcycle {i + 1}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

output_path = BASE_DIR / "detected_images" / "motorcycle_test.jpg"

cv2.imwrite(str(output_path), annotated)

print("Result saved:", output_path)

# Check overlap between motorcycle detections
print("\nChecking bounding box overlaps:")

boxes = results[0].boxes.xyxy.cpu().numpy()

for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):

        x_left = max(boxes[i][0], boxes[j][0])
        y_top = max(boxes[i][1], boxes[j][1])
        x_right = min(boxes[i][2], boxes[j][2])
        y_bottom = min(boxes[i][3], boxes[j][3])

        intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])

        union = area_i + area_j - intersection

        iou = intersection / union if union > 0 else 0

        print(f"Motorcycle {i + 1} and {j + 1}: IoU = {iou:.3f}")



