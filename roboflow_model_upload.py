from roboflow import Roboflow

rf = Roboflow(api_key="OJf6nkwQVafiiN9Rr2l8")
workspace = rf.workspace("busaps")

workspace.deploy_model(
  model_type="yolov11",
  model_path="./trained_models",
  filename="latest-20250717134940-tuned.pt",
  project_ids=["bus-aps-det"],
  model_name="bus-aps-det-yolov11"
)