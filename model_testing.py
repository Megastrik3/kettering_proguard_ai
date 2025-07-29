from ultralytics import YOLO
import os
import model_metrics

if __name__ == "__main__":
  selected_model = model_metrics.getModels()

  model = YOLO(selected_model)

  metrics = model.val(data="./datasets/bus-aps/data.yaml", split="val", save_json=True, save=True, imgsz=640, device=0, verbose=True, save_txt=True, plots=True, visualize=True)
  print(metrics.box.map)
