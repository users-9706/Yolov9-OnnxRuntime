    #Export onnx by ultralytics
    from ultralytics import YOLO 
    model = YOLO('yolov9c.pt')
    model.export(format='onnx') 
