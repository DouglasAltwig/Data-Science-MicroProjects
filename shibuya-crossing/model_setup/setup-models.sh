#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if YOLO_MODELS is set
if [ -z "$YOLO_MODELS" ]; then
  echo "Error: YOLO_MODELS environment variable is not set."
  exit 1
fi

echo "Starting model setup for: $YOLO_MODELS"

# Convert the comma-separated string into a space-separated list for the loop
for model_version in $(echo $YOLO_MODELS | sed "s/,/ /g")
do
  MODEL_DIR="/model_repository/$model_version"
  MODEL_PATH="$MODEL_DIR/1/model.onnx"
  CONFIG_PATH="$MODEL_DIR/config.pbtxt"

  # 1. Check if the model files already exist
  if [ -f "$MODEL_PATH" ] && [ -f "$CONFIG_PATH" ]; then
    echo "✔ Model '$model_version' already exists. Skipping."
  else
    echo "➤ Model '$model_version' not found. Preparing..."

    # 2. Download and export the model using the 'yolo' CLI
    # The 'yolo' command downloads the .pt file and exports it to .onnx
    echo "   - Downloading and exporting to ONNX format..."
    # yolo export model=${model_version}.pt format=onnx imgsz=640
    yolo export model=${model_version}.pt format=onnx imgsz=640 opset=17

    # 3. Create the required directory structure for Triton
    echo "   - Creating Triton directory structure: $MODEL_DIR/1/"
    mkdir -p "$MODEL_DIR/1"

    # 4. Move the exported ONNX file to the correct location
    echo "   - Moving ONNX model to $MODEL_PATH"
    mv "${model_version}.onnx" "$MODEL_PATH"

    # 5. Generate the config.pbtxt file for Triton
    echo "   - Generating Triton config.pbtxt at $CONFIG_PATH"
    cat <<EOF > "$CONFIG_PATH"
name: "$model_version"
backend: "onnxruntime"
# max_batch_size: 1
# dynamic_batching { }
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 1, 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 1, 84, 8400 ]
  }
]
EOF
    echo "✔ Successfully prepared model '$model_version'."
  fi
done

echo "Model setup complete."