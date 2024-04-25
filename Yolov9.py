import cv2
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
model_path = 'yolov9c.onnx'
start_time = cv2.getTickCount()
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
conf_thresold = 0.45
iou_threshold = 0.25
model_inputs = session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]
input_shape = model_inputs[0].shape
model_output = session.get_outputs()
output_names = [model_output[i].name for i in range(len(model_output))]
input_height, input_width = input_shape[2:]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (input_width, input_height))
input_image = resized / 255.0
input_image = input_image.transpose(2,0,1)
input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
outputs = session.run(output_names, {input_names[0]: input_tensor})[0]
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
indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thresold, nms_threshold=iou_threshold)
detections = []
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
end_time = cv2.getTickCount()
t = (end_time - start_time)/cv2.getTickFrequency()
fps = 1/t
print(f"EStimated FPS: {fps:.2f}")
cv2.putText(image, 'FPS: {:.2f}'.format(fps), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, [225, 0, 0], 2, 8);
cv2.imshow("YOLOV9-ONNXRUNTIME", image)
cv2.waitKey(0)
