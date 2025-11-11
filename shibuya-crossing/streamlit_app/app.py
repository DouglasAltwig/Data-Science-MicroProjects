import streamlit as st
import requests
import os

API_URL = os.getenv('API_URL', 'http://api:8000/process-video/')
MINIO_PUBLIC_URL = os.getenv('MINIO_PUBLIC_URL', 'http://localhost:9001/processed-videos')


st.set_page_config(layout="wide", page_title="Shibuya Crossing: Video Object Detection & Tracking")
st.title("Shibuya Crossing: Video Object Detection & Tracking")

st.sidebar.header("Configuration")
model_selection = st.sidebar.selectbox("Select YOLO Model", ["yolov12l", "yolo12m", "yolo12n", "yolov8s", "yolo12x"])
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.50, 0.05)
iou_thresh = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.60, 0.05)

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if 'last_job_id' not in st.session_state:
    st.session_state.last_job_id = None

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Process Video"):
        with st.spinner("Submitting job..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            payload = {
                "model": model_selection,
                "conf_thresh": conf_thresh,
                "iou_thresh": iou_thresh
            }
            try:
                response = requests.post(API_URL, files=files, data=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.last_job_id = result.get("job_id")
                    st.success(f"✅ Job submitted successfully! Job ID: {st.session_state.last_job_id}")
                else:
                    st.error(f"❌ Error submitting job: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: Could not connect to the API. Is it running? Details: {e}")

if st.session_state.last_job_id:
    st.markdown("---")
    st.subheader("Last Processed Job")
    st.write(f"**Job ID:** `{st.session_state.last_job_id}`")
    st.info("Processing is done in the background. Once complete, the video will be available for download.")
    
    # Construct the potential download URL
    # NOTE: This assumes the input file extension is preserved.
    # In a real system, a database would track the output filename.
    file_extension = ".mp4" # A safe default
    if uploaded_file:
         file_extension = os.path.splitext(uploaded_file.name)[1]
    
    processed_video_url = f"{MINIO_PUBLIC_URL}/{st.session_state.last_job_id}{file_extension}"
    st.markdown(f"**Download Link:** [{processed_video_url}]({processed_video_url})")
    st.warning("Note: It may take a few minutes for the processed video to appear. Please be patient and refresh if needed.")