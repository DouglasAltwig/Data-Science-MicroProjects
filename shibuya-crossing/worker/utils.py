import numpy as np
import cv2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, stride=32):
    """Resize and pad image while preserving aspect ratio"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    # Resize and pad
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=color
    )
    return im, r, (left, top)

def postprocess(
    output_data: np.ndarray,
    original_shape: tuple,
    conf_thresh: float,
    iou_thresh: float,
    ratio: float,
    pad: tuple
) -> np.ndarray:
    """
    Convert model output to detection results with proper unpadding
    Returns: np.ndarray of shape [N, 6] where columns are:
        [x1, y1, x2, y2, confidence, class_id]
    """
    # (1, 84, 8400) -> (8400, 84)
    outputs = np.squeeze(output_data).T
    
    # Filter by confidence before processing
    obj_conf = np.max(outputs[:, 4:], axis=1)
    valid_indices = obj_conf > conf_thresh
    
    if not np.any(valid_indices):
        return np.empty((0, 6))
    
    outputs = outputs[valid_indices]
    boxes = outputs[:, :4]
    scores = np.max(outputs[:, 4:], axis=1)
    class_ids = np.argmax(outputs[:, 4:], axis=1)
    
    # Unpad and scale coordinates
    left_pad, top_pad = pad
    xyxy = _convert_and_unpad(boxes, ratio, left_pad, top_pad, original_shape)
    
    # Apply NMS
    nms_indices = cv2.dnn.NMSBoxes(
        xyxy.tolist(),
        scores.tolist(),
        conf_thresh,
        iou_thresh
    )
    
    if len(nms_indices) == 0:
        return np.empty((0, 6))
    
    # Format final detections
    final_detections = np.column_stack((
        xyxy[nms_indices],
        scores[nms_indices],
        class_ids[nms_indices]
    ))
    
    return final_detections.astype(np.float32)

def _convert_and_unpad(boxes, ratio, left_pad, top_pad, orig_shape):
    """Convert YOLO format to xyxy and unpad coordinates"""
    # Convert cx,cy,w,h to xyxy in model space
    xyxy = np.zeros((boxes.shape[0], 4), dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # Unpad coordinates
    xyxy[:, [0, 2]] -= left_pad
    xyxy[:, [1, 3]] -= top_pad
    
    # Scale to original image space
    xyxy /= ratio
    
    # Clip to image boundaries
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_shape[1])
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_shape[0])
    
    return xyxy