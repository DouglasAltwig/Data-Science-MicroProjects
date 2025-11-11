import os
import sys
import json
import time
import pika
import boto3
import numpy as np
import tritonclient.http as httpclient
import logging
import tempfile
from prometheus_client import start_http_server, Counter, Gauge
import supervision as sv
from utils import letterbox, postprocess
from class_names import get_class_name, COCO_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Configuration
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq')
TRITON_URL = os.getenv('TRITON_URL', 'triton:8000')
MINIO_URL = os.getenv('MINIO_URL', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
RAW_BUCKET = 'raw-videos'
PROCESSED_BUCKET = 'processed-videos'
MODEL_INPUT_SHAPE = (640, 640)

# Prometheus Metrics
JOBS_PROCESSED = Counter('worker_jobs_processed_total', 'Total processed jobs')
JOBS_FAILED = Counter('worker_jobs_failed_total', 'Total failed jobs')
PROCESSING_FPS = Gauge('worker_processing_fps', 'Current processing FPS')

# Initialize clients
triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
s3_client = boto3.client(
    's3',
    endpoint_url=f'http://{MINIO_URL}',
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)
try:
    s3_client.head_bucket(Bucket=PROCESSED_BUCKET)
except s3_client.exceptions.ClientError as e:
    if e.response['Error']['Code'] == '404':
        logging.info(f"Creating bucket '{PROCESSED_BUCKET}'")
        s3_client.create_bucket(Bucket=PROCESSED_BUCKET)
    else:
        raise

def process_frame(frame: np.ndarray, model_name: str, conf_thresh: float, iou_thresh: float) -> sv.Detections:
    """Process single frame through Triton inference pipeline"""
    orig_shape = frame.shape[:2]
    img, ratio, pad = letterbox(frame, MODEL_INPUT_SHAPE)
    
    # Preprocess for model
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 3, *MODEL_INPUT_SHAPE)

    # Inference
    inputs = [httpclient.InferInput('images', img.shape, "FP32")]
    inputs[0].set_data_from_numpy(img)
    outputs = [httpclient.InferRequestedOutput('output0')]

    try:
        response = triton_client.infer(model_name, inputs, outputs=outputs)
        raw_detections = response.as_numpy('output0')
        results = postprocess(
            raw_detections, 
            orig_shape, 
            conf_thresh, 
            iou_thresh,
            ratio,
            pad
        )
        
        # Convert to supervision format
        if len(results) > 0:
            return sv.Detections(
                xyxy=results[:, :4],
                confidence=results[:, 4],
                class_id=results[:, 5].astype(int)
            )
        return sv.Detections.empty()
    except httpclient.InferenceServerException as e:
        logging.error(f"Inference failed: {str(e)}")
        return sv.Detections.empty()

def process_video(source_path: str, target_path: str, config: dict) -> int:
    """Process entire video with object detection and tracking"""
    video_info = sv.VideoInfo.from_video_path(source_path)
    tracker = sv.ByteTrack(frame_rate=video_info.fps)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    with sv.VideoSink(target_path, video_info) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_path), 1):
            # Get detections
            detections = process_frame(
                frame,
                config['model'],
                config['conf_thresh'],
                config['iou_thresh']
            )
            
            # Tracking and annotation
            tracked_dets = tracker.update_with_detections(detections)
            
            # Create labels using the get_class_name function
            labels = []
            for i in range(len(tracked_dets)):
                class_id = int(tracked_dets.class_id[i])
                tracker_id = int(tracked_dets.tracker_id[i])
                class_name = get_class_name(class_id)
                labels.append(f"#{tracker_id} {class_name}")
            
            annotated_frame = box_annotator.annotate(frame.copy(), tracked_dets)
            annotated_frame = label_annotator.annotate(annotated_frame, tracked_dets, labels)
            sink.write_frame(annotated_frame)
            
    return frame_idx

def callback(ch, method, properties, body):
    """RabbitMQ message handler"""
    start_time = time.time()
    msg = json.loads(body)
    job_id = msg.get('job_id')
    object_name = msg.get('object_name')
    logging.info(f"Processing job {job_id} for {object_name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, object_name)
        output_path = os.path.join(tmpdir, f"processed_{object_name}")

        try:
            # Download from MinIO
            s3_client.download_file(RAW_BUCKET, object_name, input_path)
            
            # Process video
            frame_count = process_video(
                input_path,
                output_path,
                msg
            )
            
            # Upload result
            s3_client.upload_file(output_path, PROCESSED_BUCKET, object_name)
            
            # Update metrics
            duration = time.time() - start_time
            PROCESSING_FPS.set(frame_count / duration if duration else 0)
            JOBS_PROCESSED.inc()
            logging.info(f"Job {job_id} completed: {frame_count} frames at {frame_count/duration:.1f} FPS")
            
        except Exception as e:
            JOBS_FAILED.inc()
            logging.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    logging.info("VisionFlow Worker starting...")
    start_http_server(8001)  # Prometheus metrics endpoint

    while True:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=RABBITMQ_HOST)
            )
            channel = connection.channel()
            channel.queue_declare(queue='video_processing_queue', durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(
                queue='video_processing_queue',
                on_message_callback=callback
            )
            logging.info("Waiting for messages...")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as e:
            logging.warning(f"RabbitMQ connection failed: {e}. Retrying in 5s")
            time.sleep(5)
        except KeyboardInterrupt:
            logging.info("Shutting down gracefully...")
            break
        except Exception as e:
            logging.exception("Unexpected error")
            time.sleep(5)
        finally:
            if 'connection' in locals() and connection.is_open:
                connection.close()

if __name__ == '__main__':
    main()
