from ultralytics import YOLO

model = YOLO("C:\\Users\\megas\\Documents\\Python\\Kettering_REU\\YOLO\\trained_models\\latest-20250722135202.onnx", task="detect")

results = model.predict(source="./test3.mp4", show=True, save=True, save_txt=False, conf=0.55, iou=0.45, device=0, stream=True, show_boxes=True, show_labels=True, show_conf=True)

for r in results:
    print(r.probs)