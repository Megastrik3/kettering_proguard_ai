from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("./trained_models/latest-20250717134940.engine")  # load an official model

    # Validate the model
    metrics = model.val(data="datasets/bus-aps/data.yaml", imgsz=640, device=0, verbose=True, save=True)  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95(M)
    metrics.box.map50  # map50(M)
    metrics.box.map75  # map75(M)
    metrics.box.maps  # a list contains map50-95(M) of each category

if __name__ == "__main__":
    main()