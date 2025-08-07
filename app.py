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

st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("Object Detection Web using YOLOv8")
st.write("Tải lên một bức ảnh và AI sẽ phát hiện các đối tượng có trong đó")
uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Kết quả Nhận diện")
        with st.spinner('Đang xử lý...'):
            result_image = detect_objects(image)
            st.image(result_image, use_container_width=True)