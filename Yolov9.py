import cv2
import time
import numpy as np
import onnxruntime as ort
def readClassesNames(file_path):
    with open(file_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
classes_names = 'coco.names'
classes = readClassesNames(classes_names)
image = cv2.imread('bus.jpg')
image_height, image_width = image.shape[:2]
model_path = 'yolov9-c.onnx'
start_time = time.time()
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
conf_thresold = 0.45
iou_threshold = 0.25
model_inputs = session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]
input_shape = model_inputs[0].shape
model_output = session.get_outputs()
output_names = [model_output[i].name for i in range(len(model_output))]
input_height, input_width = input_shape[2:]
#获取输入
resized = cv2.resize(image, (input_width, input_height))
input_image = resized / 255.0
input_image = input_image.transpose(2, 0, 1)
input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
#获取输出
outputs = session.run(output_names, {input_names[0]: input_tensor})[0]
#输出处理
predictions = np.squeeze(outputs).T
scores = np.max(predictions[:, 4:], axis=1)
predictions = predictions[scores > conf_thresold, :]
scores = scores[scores > conf_thresold]
class_ids = np.argmax(predictions[:, 4:], axis=1)
boxes = predictions[:, :4]
input_shape = np.array([input_width, input_height, input_width, input_height])
boxes = np.divide(boxes, input_shape, dtype=np.float32)
boxes *= np.array([image_width, image_height, image_width, image_height])
boxes = boxes.astype(np.int32)
def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes
def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    iou = intersection_area / union_area
    return iou
indices = nms(boxes, scores, iou_threshold)
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
    bbox = bbox.round().astype(np.int32).tolist()
    cls_id = int(label)
    cls = classes[cls_id]
    cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), (0,0,255), 2, 8)
    cv2.rectangle(image, (bbox[0], (bbox[1]-20)), (bbox[2], bbox[1]), (0,255,255), -1)
    cv2.putText(image, f'{cls}', (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_PLAIN,2, [225, 0, 0], thickness=2)
end_time = time.time()
fps = 1/(end_time - start_time)
print(f"EStimated FPS: {fps:.2f}")
cv2.putText(image, 'FPS: {:.2f}'.format(fps), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, [225, 0, 0], 2, 8);
cv2.imshow("YOLOV9-ONNXRUNTIME", image)
cv2.waitKey(0)
