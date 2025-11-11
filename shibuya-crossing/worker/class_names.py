"""
Class names configuration for object detection models.
You can modify this file to match your specific model's classes.
"""

# COCO dataset classes (80 classes) - Used by YOLOv5, YOLOv8, YOLOv11, etc.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def get_class_name(class_id: int, class_names: list = None) -> str:
    """
    Get class name for a given class ID.
    
    Args:
        class_id: Integer class ID from model prediction
        class_names: Optional custom class names list. If None, uses COCO_CLASSES
        
    Returns:
        Class name string, or "class_{id}" if ID is out of range
    """
    if class_names is None:
        class_names = COCO_CLASSES
    
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"