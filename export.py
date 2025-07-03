from ultralytics import YOLO

model = YOLO("best-seg.pt")
output = model.export(format="imx", data="datasets/bus-aps/data.yaml")