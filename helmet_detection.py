
from pathlib import Path
import cv2
from ultralytics import YOLO

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load trained helmet detection model
MODEL_PATH = BASE_DIR / "models" / "helmet_model.pt"

model = YOLO(str(MODEL_PATH))


def detect_helmets(image_path):

    # Run YOLO detection
    results = model.predict(
    source=str(image_path),
    conf=0.20,
    imgsz=960,
    verbose=False
)

    # Draw bounding boxes
    annotated_image = results[0].plot()
    print("\nTotal detections:", len(results[0].boxes))

    for box in results[0].boxes:
        print(
            "Class:", model.names[int(box.cls[0])],
            "| Confidence:", round(float(box.conf[0]), 3),
            "| Bounding box:", box.xyxy[0].tolist()
        )

    return annotated_image, results[0]


if __name__ == "__main__":

    # Input image
    image_path = BASE_DIR / "uploads" / "test.jpeg"

    # Output image
    output_path = BASE_DIR / "detected_images" / "result.jpeg"

    print("Starting helmet detection...")

    # Check image exists
    if not image_path.exists():
        print("Error: Test image not found!")
    else:

        # Run detection
        image, results = detect_helmets(image_path)

        # Save processed image
        cv2.imwrite(str(output_path), image)

        print("\nDetection completed!")

        # Display detected classes
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(
                f"{model.names[class_id]} "
                f"- Confidence: {confidence:.2f}"
            )

        print("\nResult saved at:", output_path)


