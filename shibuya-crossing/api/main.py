import os
import uuid
import pika
import boto3
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

app = FastAPI()

# --- Configuration from Environment Variables ---
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq')
MINIO_HOST = os.getenv('MINIO_URL', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
RAW_BUCKET = 'raw-videos'

# --- Service Connections ---
try:
    s3_client = boto3.client(
        's3',
        endpoint_url=f'http://{MINIO_HOST}',
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )
    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=RAW_BUCKET)
    except s3_client.exceptions.ClientError:
        s3_client.create_bucket(Bucket=RAW_BUCKET)
except Exception as e:
    print(f"Error connecting to MinIO: {e}")
    s3_client = None


def get_rabbitmq_connection():
    return pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))

@app.post("/process-video/")
async def process_video(
    file: UploadFile = File(...),
    model: str = Form("yolov8n"),
    conf_thresh: float = Form(0.25),
    iou_thresh: float = Form(0.45),
):
    if not s3_client:
        raise HTTPException(status_code=503, detail="Storage service is unavailable.")

    job_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    object_name = f"{job_id}{file_extension}"

    # 1. Upload video to MinIO
    try:
        s3_client.upload_fileobj(file.file, RAW_BUCKET, object_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file to storage: {e}")

    # 2. Push job to RabbitMQ
    job_payload = {
        "job_id": job_id,
        "object_name": object_name,
        "model": model,
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
    }

    try:
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        channel.queue_declare(queue='video_processing_queue', durable=True)
        channel.basic_publish(
            exchange='',
            routing_key='video_processing_queue',
            body=json.dumps(job_payload),
            properties=pika.BasicProperties(delivery_mode=2) # make message persistent
        )
        connection.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to queue job: {e}")

    return {"message": "Job submitted successfully", "job_id": job_id}

@app.get("/health")
def health_check():
    return {"status": "ok"}