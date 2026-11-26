
import streamlit as st
import cv2
import easyocr
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "detected_images"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    helmet_model = YOLO(
        str(MODEL_DIR / "helmet_model.pt")
    )

    plate_model = YOLO(
        str(MODEL_DIR / "plate_model.pt")
    )

    reader = easyocr.Reader(
        ['en'],
        gpu=False
    )

    return helmet_model, plate_model, reader


# ==========================================
# IMAGE PROCESSING
# ==========================================

def process_frame(frame, helmet_model, plate_model, reader):

    # Helmet detection
    helmet_results = helmet_model.predict(
        source=frame,
        conf=0.20,
        imgsz=960,
        verbose=False
    )

    annotated = helmet_results[0].plot()

    helmet_counts = {
        "With Helmet": 0,
        "Without Helmet": 0
    }

    for box in helmet_results[0].boxes:

        class_id = int(box.cls[0])
        class_name = helmet_model.names[class_id]

        if class_name in helmet_counts:
            helmet_counts[class_name] += 1

    # Number plate detection
    plate_results = plate_model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    detected_plates = []

    for box in plate_results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Keep coordinates inside the image
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        # Preprocess plate image
        gray = cv2.cvtColor(
            plate_crop,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.bilateralFilter(
            gray,
            11,
            17,
            17
        )

        # OCR recognition
        text = reader.readtext(
            gray,
            detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        registration = "".join(text).replace(" ", "").upper()

        detected_plates.append(registration)

        # Draw plate bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            registration if registration else "Plate detected",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return annotated, helmet_counts, detected_plates


# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

st.title("Helmet Detection and Number Plate Recognition System")

st.write(
    "An AI-based prototype for detecting helmet status "
    "and recognising vehicle registration numbers."
)

st.divider()

st.subheader("Upload Image or Video")

uploaded_file = st.file_uploader(
    "Choose a road image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)


# ==========================================
# RUN DETECTION
# ==========================================

if uploaded_file is not None:

    file_path = UPLOAD_DIR / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    if st.button("Run Detection", type="primary"):

        try:

            with st.spinner("Loading AI models..."):

                helmet_model, plate_model, reader = load_models()

            # ----------------------------------
            # IMAGE PROCESSING
            # ----------------------------------

            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

                image = cv2.imread(str(file_path))

                if image is None:
                    st.error("Unable to read the uploaded image.")
                    st.stop()

                with st.spinner("Detecting helmets and number plates..."):

                    result, helmet_counts, plates = process_frame(
                        image,
                        helmet_model,
                        plate_model,
                        reader
                    )

                output_path = OUTPUT_DIR / "result.jpg"
                cv2.imwrite(str(output_path), result)

                st.subheader("Detection Result")

                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

                # Summary
                st.subheader("Detection Summary")

                col1, col2 = st.columns(2)

                col1.metric(
                    "With Helmet",
                    helmet_counts["With Helmet"]
                )

                col2.metric(
                    "Without Helmet",
                    helmet_counts["Without Helmet"]
                )

                st.write("**Recognised Registration Numbers:**")

                if plates:
                    for plate in plates:
                        st.write(plate if plate else "Unreadable plate")
                else:
                    st.warning("No number plates detected.")

                st.caption(
                    "These are model predictions, not confirmed violations. "
                    "Please verify uncertain results manually."
                )

            # ----------------------------------
            # VIDEO PROCESSING
            # ----------------------------------

            else:

                cap = cv2.VideoCapture(str(file_path))

                if not cap.isOpened():
                    st.error("Unable to open the uploaded video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps <= 0:
                    fps = 25

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_path = OUTPUT_DIR / "processed_video.mp4"

                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                progress = st.progress(0)
                status = st.empty()

                total_helmet_counts = {
                    "With Helmet": 0,
                    "Without Helmet": 0
                }

                all_plates = set()
                frame_number = 0

                with st.spinner("Processing video..."):

                    while True:

                        ret, frame = cap.read()

                        if not ret:
                            break

                        result, counts, plates = process_frame(
                            frame,
                            helmet_model,
                            plate_model,
                            reader
                        )

                        writer.write(result)

                        for key in total_helmet_counts:
                            total_helmet_counts[key] += counts[key]

                        for plate in plates:
                            if plate:
                                all_plates.add(plate)

                        frame_number += 1

                        if total_frames > 0:
                            progress.progress(
                                min(frame_number / total_frames, 1.0)
                            )

                        status.write(
                            f"Processed {frame_number} frames"
                        )

                cap.release()
                writer.release()

                st.success("Video processing completed!")

                st.subheader("Processed Video")

                st.video(str(output_path))

                st.subheader("Detection Summary")

                col1, col2 = st.columns(2)

                col1.metric(
                    "With Helmet detections",
                    total_helmet_counts["With Helmet"]
                )

                col2.metric(
                    "Without Helmet detections",
                    total_helmet_counts["Without Helmet"]
                )

                st.write("**Recognised Registration Numbers:**")

                if all_plates:
                    for plate in sorted(all_plates):
                        st.write(plate)
                else:
                    st.warning("No number plates detected.")

                st.caption(
                    "Video counts are detection totals across frames, "
                    "not unique rider counts. Verify results manually."
                )

        except Exception as error:
            st.error(f"An error occurred: {error}")

