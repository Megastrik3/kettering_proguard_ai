"""
This script is used to generate model metrics using the test dataset specifically.
User's will first be prompted to select a model from the list of available models.
"""
from ultralytics import YOLO
import os
import main

if __name__ == "__main__":
  selected_model = main.getModels()

  model = YOLO(selected_model)

  metrics = model.val(data="./datasets/bus-aps/data.yaml", split="test", save_json=True, save=True, imgsz=640, device=0, verbose=True, save_txt=True, plots=True, visualize=True)
  print(metrics.box.map)
