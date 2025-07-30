import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.types as fot
import os

# --- Configuration ---

# 1. Define the classes to keep
TARGET_CLASSES = ["person", "backpack"]

# 2. Set the maximum number of images to download
MAX_SAMPLES = 2000

# 3. Define the export format and output directory
# You can change fot.COCODetectionDataset to fot.YOLOv5Dataset, etc.
EXPORT_FORMAT = fot.COCODetectionDataset
EXPORT_DIR = os.path.join(os.getcwd(), "exported-coco-dataset")

# 4. Set the name for the dataset in FiftyOne
DATASET_NAME = f"coco-2017-people-backpacks-clean-{MAX_SAMPLES}"


def download_clean_and_export_coco():
    """
    Downloads, cleans, and exports data from the COCO-2017 dataset.

    This function will:
    1. Load images from COCO that contain 'person' or 'backpack'.
    2. Limit the download to a maximum of 2000 images.
    3. Manually iterate through each sample to remove unwanted labels and all
       segmentation data, leaving only bounding boxes for the target classes.
    4. Export the cleaned dataset to the specified local directory.
    5. Launch the FiftyOne App to visualize the final result.
    """
    # --- Stage 1: Initial Download ---
    print(f"Downloading a maximum of {MAX_SAMPLES} images containing: {', '.join(TARGET_CLASSES)}...")

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train", "validation"],
        label_types=["detections"],
        classes=TARGET_CLASSES,
        max_samples=MAX_SAMPLES,
        dataset_name=DATASET_NAME,
    )

    # --- Stage 2: Clean Up Labels and Remove Segmentations ---
    print("\nDownload complete. Now cleaning up labels and removing segmentation data...")

    ground_truth_field = "ground_truth"
    for sample in dataset.iter_samples(autosave=True):
        if sample[ground_truth_field] is None:
            continue

        # Keep only the detections whose label is in our target class list
        filtered_detections = [
            d for d in sample[ground_truth_field].detections if d.label in TARGET_CLASSES
        ]

        # Explicitly remove any segmentation data from the kept detections
        for detection in filtered_detections:
            detection.mask = None

        sample[ground_truth_field].detections = filtered_detections

    print("Cleanup complete.")
    dataset.persistent = True

    # --- Stage 3: Export the Cleaned Dataset ---
    print(f"\nExporting dataset to '{EXPORT_DIR}' in COCO format...")

    # Ensure the export directory exists
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

    dataset.export(
        export_dir=EXPORT_DIR,
        dataset_type=EXPORT_FORMAT,
        label_field=ground_truth_field,
    )
    print("Export complete! ✅")


    # --- Stage 4: Visualize the Result ---
    print("\nDataset summary:")
    print(dataset)

    session = fo.launch_app(dataset)
    print("\nFiftyOne App launched. You can now explore the clean dataset. 🚀")
    session.wait()


if __name__ == "__main__":
    download_clean_and_export_coco()