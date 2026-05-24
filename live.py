import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

# Load YOLO model
model = YOLO("best.pt")

# Streamlit page
st.set_page_config(
    page_title="Live Webcam Posture Detection",
    page_icon="🪑",
    layout="centered"
)

st.title("Live Webcam Posture Detection")
st.write("Real-time posture detection using your webcam.")

# Confidence slider
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Video Processor
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Run YOLO prediction
        results = model.predict(
            source=img,
            conf=confidence,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Add confidence labels
        boxes = results[0].boxes

        for box in boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf_score = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{class_name}: {conf_score:.2f}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )

# Start webcam stream
webrtc_streamer(
    key="posture-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)