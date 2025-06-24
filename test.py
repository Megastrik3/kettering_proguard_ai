from ultralytics import YOLO


def main():
    print("Starting YOLO Training...")
    model = YOLO("best-seg.pt") 
    output = model.export(format="engine")
    print("Export completed.")

if __name__ == '__main__':
    main()
