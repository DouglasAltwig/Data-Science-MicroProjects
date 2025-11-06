import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

print("Starting the Object Detection and Tracking script...")

# Load the YOLO model
model = YOLO("yolo12x.pt")
print("YOLOv8 model loaded successfully.")

# Initialize the ByteTrack tracker from the supervision library
tracker = sv.ByteTrack()
print("ByteTrack tracker initialized.")

# Initialize annotators for drawing boxes and labels
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Path to the input and output videos
input_video_path = "shibuya-crossing-input.mp4"
output_video_path = "shibuya-crossing-output.mp4"

# Open the video file for reading
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
    exit()

# Get video properties for the output writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
print(f"Output video will be saved to {output_video_path}")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO inference on the frame
    results = model.predict(
        source=frame,
        imgsz=1280,
        augment=True,
        agnostic_nms=True,
        verbose=False,
    )[0]

    # Convert YOLO results to a supervision Detections object
    detections = sv.Detections.from_ultralytics(results)

    # Filter detections (optional, e.g., only track 'person' class)
    # detections = detections[detections.class_id == 0]

    # Update the tracker with the new detections
    tracked_detections = tracker.update_with_detections(detections)

    # Prepare labels for annotation
    labels = [
        f"#{tracker_id} {model.model.names[class_id]}"
        for class_id, tracker_id in zip(
            tracked_detections.class_id, tracked_detections.tracker_id
        )
    ]

    # Annotate the frame with bounding boxes and labels
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=tracked_detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=tracked_detections, labels=labels
    )

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")


# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Script finished successfully. Output saved.")
