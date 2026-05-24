import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Posture Detection using YOLOv8",
    page_icon="🪑",
    layout="centered"
)

st.title("Posture Detection using YOLOv8")
st.write("Upload an image to detect posture classes.")

# Load model
model = YOLO("best.pt")

# Confidence slider
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run prediction
    results = model.predict(
        source=temp_path,
        conf=confidence,
        save=False
    )

    # Plot results
    annotated_image = results[0].plot()

    st.subheader("Detection Result")
    st.image(annotated_image, channels="BGR", use_container_width=True)

    # Show detections
    st.subheader("Detected Classes")

    boxes = results[0].boxes

    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf_score = float(box.conf[0])

            st.write(f"Class: {class_name} | Confidence: {conf_score:.2f}")
    else:
        st.write("No posture detected.")

    # Remove temp file
    os.remove(temp_path)