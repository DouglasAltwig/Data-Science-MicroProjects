#!/bin/sh

# Pro-tip: 'set -e' will make the script exit immediately if a command fails.
# This would have prevented the misleading "success" message from printing.
set -e

# Configure the MinIO client (mc) alias
mc alias set myminio http://minio:9000 minioadmin minioadmin

# Use 'mc ready' to wait for MinIO to be online
until mc ready myminio; do
  echo "Waiting for MinIO..."
  sleep 2
done

echo "MinIO is ready."

# Create buckets if they don't exist
mc mb myminio/raw-videos || true
mc mb myminio/processed-videos || true
echo "Buckets are ready."

# Set the anonymous (public) policy using the correct 'set-json' subcommand
mc anonymous set-json /app/public_policy.json myminio/processed-videos
echo "Anonymous download policy set for 'processed-videos' bucket."

echo "MinIO setup complete."