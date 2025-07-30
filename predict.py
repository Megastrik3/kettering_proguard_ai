from ultralytics import YOLO
import main

if __name__ == "__main__":
    selected_model = main.getModels()

    model = YOLO(selected_model)

    results = model.predict("./datasets/bus-aps/test/images", show=False, save=True, save_txt=False, conf=0.55, iou=0.45, device=0, stream=False, show_boxes=True, show_labels=True, show_conf=True)

    for r in results:
        print(r.probs)