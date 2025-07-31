"""
This script deploys the exported YOLO model to the Sony IMX500 camera and starts live inference on the video stream.
Code was taken and adapted from the documentation found here: https://docs.ultralytics.com/integrations/sony-imx500/
"""
import numpy as np
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics


class YOLO(Model):
    """YOLO model for IMX500 deployment."""
    def __init__(self, model_path):
        """Initialize the YOLO model for IMX500 deployment."""
        super().__init__(
            model_file=f'{model_path}/packerOut.zip',
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

        self.labels = np.genfromtxt(
            f"{model_path}/labels.txt",  # replace with proper directory
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        """Post-process the output tensors for object detection."""
        return pp_od_yolo_ultralytics(output_tensors)


def main(model_path='trained_models'):
    device = AiCamera(frame_rate=12)  # Optimal frame rate for maximum DPS of the YOLO model running on the AI Camera
    model = YOLO(model_path)
    device.deploy(model)

    dps_counter = 0
    dps_total = 0
    fps_counter = 0
    fps_total = 0
    annotator = Annotator()
    delay_counter = 0
    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.55]
            labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]

            annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
            metrics = frame.display() 
            delay_counter = delay_counter + 1
            if delay_counter > 150:
                dps_counter = dps_counter + 1
                dps_total = dps_total + metrics[0]
                fps_counter = fps_counter + 1
                fps_total = fps_total + metrics[1]
                print(f'Average DPS: {(dps_total/dps_counter):.2f}')
                print(f'Average FPS: {(fps_total/fps_counter):.2f}')