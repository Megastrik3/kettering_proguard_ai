"""
This script downloads the Open Images v7 dataset, filters it for specific classes,
and exports it in COCO format for further use in machine learning tasks.
"""
import fiftyone
dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v7",
              split="test",
              label_types=["detections"],
              classes=["Backpack"],
              max_samples=2000,
          )

# Export the dataset
# Filter the dataset to only include "Backpack" detections
filtered_view = dataset.filter_labels(
    "ground_truth",
    (fiftyone.ViewField("label") == "Backpack") | (fiftyone.ViewField("label") == "Person")
)

filtered_view.export(
    export_dir=r".\datasets",
    dataset_type=fiftyone.types.dataset_types.COCODetectionDataset,
    label_field="ground_truth",
)

if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
    session = fiftyone.launch_app(dataset)
    session.wait()