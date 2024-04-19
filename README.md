# Yolov9-OnnxRuntime
git clone https://github.com/WongKinYiu/yolov9.git
export onnx
python export.py --weights yolov9-c.pt --simplify --include "onnx"
