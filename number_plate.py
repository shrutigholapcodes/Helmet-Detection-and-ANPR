

from ultralytics import YOLO
from pathlib import Path
import cv2
import easyocr

BASE_DIR = Path(__file__).resolve().parent

# Load number plate model
MODEL_PATH = BASE_DIR / "models" / "plate_model.pt"
model = YOLO(str(MODEL_PATH))

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)


def detect_number_plate(image_path):

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    image = cv2.imread(str(image_path))
    detected_plates = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        plate_crop = image[y1:y2, x1:x2]

        # Preprocess plate
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # OCR
        text = reader.readtext(
            gray,
            detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        registration = "".join(text).replace(" ", "").upper()

        detected_plates.append(registration)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            registration,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return image, detected_plates


if __name__ == "__main__":

    image_path = BASE_DIR / "uploads" / "test.jpeg"

    image, plates = detect_number_plate(image_path)

    output_path = BASE_DIR / "detected_images" / "plate_result.jpg"

    cv2.imwrite(str(output_path), image)

    print("Detected number plates:", plates)
    print("Result saved:", output_path)
