from ultralytics import YOLO
model = YOLO("./yolo_train/runs/detect/train2/weights/best.pt")
model.export(format="onnx")