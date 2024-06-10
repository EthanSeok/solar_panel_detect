from ultralytics import YOLO
import cv2
model = YOLO("./yolo_train/runs/detect/train2/weights/best.pt")

result = model.predict("./test_images/test14.jpg", save=True, conf=0.5)
plots = result[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()