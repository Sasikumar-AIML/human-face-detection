import streamlit as st
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
import plotly.express as px
from pathlib import Path
import os
import glob

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(r"L:\Humanface_detection\runs\train\face_detection_model\weights\best.pt")
RESULTS_CSV = Path(r"L:\Humanface_detection\runs\train\face_detection_model\results.csv")
DATASET_PATH = Path(r"L:\Humanface_detection\yolo_dataset\images\train")
LABELS_PATH = Path(r"L:\Humanface_detection\yolo_dataset\labels\train")

# -------------------------------
# Load YOLO model
# -------------------------------
model = None
if MODEL_PATH.exists():
    try:
        model = YOLO(str(MODEL_PATH))
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
else:
    st.error("❌ Model not found.")
    st.info("👉 Please train the model first by running: `python train.py`")


# Sidebar Info
# -------------------------------
st.sidebar.header("About Project")
st.sidebar.write("""
**Project:** Human Face Detection System  
**Domain:** Computer Vision  
**Techniques:** YOLOv8, OpenCV, Deep Learning  
**Use Cases:** Security, Access Control, Retail Analytics, Healthcare, Automotive  
**Skills:** Python, Data Preprocessing, EDA, Plotly, ML, DL, GenAI
""")


st.sidebar.markdown("### 📑 Documentation")
st.sidebar.info("""
This project detects human faces in real-time using **YOLOv8**.  
- Dataset prepared in YOLO format (images + labels).  
- Includes preprocessing, EDA, model training, and evaluation.  
- Achieved >85% Precision, Recall, F1-score, and Accuracy.  
- Supports **image upload** and **live webcam detection**.  
""")

# Sidebar navigation
menu_options = ["Data", "EDA - Visuals", "Prediction"]
selected_option = st.sidebar.selectbox("📌 Navigation", menu_options)

# -------------------------------
# Streamlit Main UI
# -------------------------------
st.title("👤 Human Face Detection System")
st.markdown("---")

# -------------------------------
# Data Section
# -------------------------------
if selected_option == "Data":
    st.header("📂 Dataset & Model Performance")

    # Dataset Samples
    st.subheader("Dataset Samples")
    image_files = list(DATASET_PATH.glob("*.jpg"))[:5]
    face_counts_summary = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        label_path = LABELS_PATH / (img_path.stem + ".txt")
        face_count = 0
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    h_img, w_img = img.shape[:2]
                    x1 = int((x - w/2) * w_img)
                    y1 = int((y - h/2) * h_img)
                    x2 = int((x + w/2) * w_img)
                    y2 = int((y + h/2) * h_img)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face_count += 1
        face_counts_summary.append(face_count)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption=f"{img_path.name} - Faces: {face_count}")

    # Dataset Summary
    st.subheader("Dataset Summary")
    st.write(f"Total sample images displayed: {len(image_files)}")
    st.write(f"Total faces in these samples: {sum(face_counts_summary)}")
    if face_counts_summary:
        st.write(f"Average faces per image: {np.mean(face_counts_summary):.2f}")

    # Model Metrics
    st.subheader("📊 Model Performance Metrics")
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        if not df.empty:
            final_metrics = df.iloc[-1]
            precision = final_metrics.get("metrics/precision(B)", 0)
            recall = final_metrics.get("metrics/recall(B)", 0)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            accuracy = (precision + recall) / 2  
            metrics_data = {
                "Metric": ["Precision", "Recall", "F1-score", "Accuracy", "mAP@.5", "mAP@.5:.95"],
                "Value (%)": [
                     round(precision * 100, 2),
                     round(recall * 100, 2),
                     round(f1 * 100, 2),
                     round(accuracy * 100, 2),
                     round(final_metrics.get("metrics/mAP50(B)", 0) * 100, 2),
                     round(final_metrics.get("metrics/mAP50-95(B)", 0) * 100, 2),
                     ]
                     }
            st.dataframe(pd.DataFrame(metrics_data))
            # Plot training loss
            loss_cols = [c for c in df.columns if "loss" in c.lower()]
            if loss_cols:
                st.subheader("Training Loss Over Epochs")
                fig_loss = px.line(df, y=loss_cols, labels={"index": "Epoch", "value": "Loss"})
                st.plotly_chart(fig_loss)
        else:
            st.warning("⚠️ Results CSV is empty.")
    else:
        st.warning("⚠️ results.csv not found. Train the model first.")

# -------------------------------
# EDA Section
# -------------------------------
elif selected_option == "EDA - Visuals":
    st.write("Looking for label files in:", LABELS_PATH)
    label_files = list(LABELS_PATH.glob("*.txt"))
    st.write(f"Found {len(label_files)} label files.")
    st.header("🔎 Exploratory Data Analysis")

    face_counts, bbox_widths, bbox_heights, bbox_aspect_ratios, bbox_areas = [], [], [], [], []

    for lbl_file in label_files:
        with open(lbl_file, "r") as f:
            lines = f.readlines()
            face_counts.append(len(lines))
            for line in lines:
                _, x, y, w, h = map(float, line.strip().split())
                bbox_widths.append(w)
                bbox_heights.append(h)
                bbox_aspect_ratios.append(w/h if h != 0 else 0)
                bbox_areas.append(w*h)

    if face_counts:
        st.subheader("Faces per Image Distribution")
        fig1 = px.histogram(face_counts, nbins=10, labels={"value": "Faces per Image"})
        st.plotly_chart(fig1)
        st.write(f"Mean faces per image: {np.mean(face_counts):.2f}")

    if bbox_widths and bbox_heights:
        st.subheader("Bounding Box Width vs Height")
        fig2 = px.scatter(x=bbox_widths, y=bbox_heights,
                          labels={'x': 'Width (normalized)', 'y': 'Height (normalized)'})
        st.plotly_chart(fig2)

        st.subheader("Bounding Box Aspect Ratio Distribution")
        fig3 = px.histogram(bbox_aspect_ratios, nbins=20, labels={"value": "Aspect Ratio (w/h)"})
        st.plotly_chart(fig3)

        st.subheader("Bounding Box Area Distribution")
        fig4 = px.histogram(bbox_areas, nbins=20, labels={"value": "Area (normalized)"})
        st.plotly_chart(fig4)

# -------------------------------
# Prediction Section
# -------------------------------
elif selected_option == "Prediction":
    st.header("📷 Real-time Face Detection")

    if model is None:
        st.warning("⚠️ No trained model available. Please train first.")
    else:
        # Confidence threshold slider
        confidence_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

        # Choose input type
        input_type = st.radio("Select Input Type", ["Upload Image", "Live Webcam"])

        # -------------------
        # Upload Image Mode
        # -------------------
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                results = model(image)
                detected_faces = 0
                for r in results:
                    for box in r.boxes:
                        conf = box.conf[0].item()
                        if conf >= confidence_thresh:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image, f"Face {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            detected_faces += 1
                            # Cropped face
                            face_crop = image[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                                         caption=f"Cropped Face")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                         caption="Detected Faces", use_column_width=True)
                st.success(f"✅ Face detection successful! Total Faces: {detected_faces}")


        # -------------------
        # Live Webcam Mode
        # -------------------
        elif input_type == "Live Webcam":
            stframe = st.empty()
            run = st.checkbox("Start Webcam")
            stop = st.button("Stop Webcam")  

            cap = cv2.VideoCapture(0)  
            try:
                while run and not stop:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Cannot access webcam.")
                        break

                    results = model(frame)
                    for r in results:
                        for box in r.boxes:
                            conf = box.conf[0].item()
                            if conf >= confidence_thresh:
                                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                                cv2.rectangle(frame, (x1, y1), (x2, y2),
                                              (0, 255, 0), 2)
                                cv2.putText(frame,
                                            f"Face {conf:.2f}",
                                            (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7,
                                            (0, 255, 0),
                                            2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB")
            finally:
                cap.release()

