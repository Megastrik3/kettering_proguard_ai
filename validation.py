from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("runs/segment/train/weights/best.pt")  # load an official model

    # Validate the model
    metrics = model.val(data="datasets/bus-aps/data.yaml", imgsz=640, device=0, verbose=True)  # no arguments needed, dataset and settings remembered
    metrics.seg.map  # map50-95(M)
    metrics.seg.map50  # map50(M)
    metrics.seg.map75  # map75(M)
    metrics.seg.maps  # a list contains map50-95(M) of each category

if __name__ == "__main__":
    main()