from ultralytics import YOLO

model = YOLO("./trained_models/latest-202507211327.engine", task="detect")

results = model.predict(source="./test.mp4", show=True, save=True, save_txt=True, conf=0.55, iou=0.45, device=0, stream=True, show_boxes=True, show_labels=True, show_conf=True)

for r in results:
    print(r.probs)