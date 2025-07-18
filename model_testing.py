from ultralytics import YOLO
import os
import model_metrics


selected_model = model_metrics.getModels()

model = YOLO(selected_model)

metrics = model.val(data="./datasets/bus-aps/data.yaml", save_json=True, split="test")
