# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    st.error(f"loi tai model")
    model = None

def detect_objects(image):

    if model is None:
        return image # Tra anh goc neu bi loi model

    # convert RGB sang BGR
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model(img_bgr)[0]

    # bounding box va label
    for box in results.boxes:
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{class_name} {conf:.2f}'
        cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # convert lai BGR sang RGB  
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)