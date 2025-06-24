from ultralytics import YOLO

# --- IMPORTANT: Point this to the exact .pt file you are using for the export ---
# For example, if you run 'yolo export model=yolov8n.pt', use 'yolov8n.pt' here.
# If you use 'yolo export model=runs/detect/train/weights/best.pt', use that full path.
PT_MODEL_PATH = 'best.pt' #<-- CHANGE THIS
# ---

try:
    print(f"Loading model from: {PT_MODEL_PATH}")
    model = YOLO(PT_MODEL_PATH)
    
    # Get the model's internal data dictionary
    model_data = model.data
    
    # Get the number of classes from the model's configuration
    num_classes = model.model.nc
    
    # Get the task type
    task = model.task
    
    print("\n" + "="*50)
    print("MODEL SELF-REPORTED PROPERTIES:")
    print(f"  - Task Type: {task}")
    print(f"  - Number of Classes (nc): {num_classes}")
    
    if 'names' in model_data:
        class_names = model_data['names']
        print(f"  - Class Names Found: {class_names}")
        if len(class_names) != num_classes:
            print(f"  - !! WARNING: Number of classes ({num_classes}) does not match number of names ({len(class_names)})!")
    else:
        print("  - !! WARNING: Could not find class names embedded in the model.")
    
    print("="*50 + "\n")

except Exception as e:
    print(f"\nAn error occurred while trying to load the .pt model: {e}")
    print("Please ensure the path is correct and the file is not corrupted.")