import math
import time
from ultralytics import YOLO
# https://www.reddit.com/r/learnpython/comments/zxxsal/open_cv_video_from_webcam_takes_abnormally_long/
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

def main():
    print("Starting YOLO Object Detection...")
    cap = cv2.VideoCapture(0)
    # Set frame time baseline
    old_frame_time = 0
    new_frame_time = 0
    # Set camera resolution
    cap.set(3, 1920)
    cap.set(4, 1080)
    # Load custom model
    # if gpu_verify.check_gpu():
    #     model_file = "busaps.engine"
    # else:
    #     model_file = "busaps.onnx"
    model_file = "best-seg.pt"  # Use the ONNX model for CPU inference
    base_model = YOLO(model_file)
    export = base_model.export(format='ncnn')
    model = YOLO(export)  # Load the exported model
    print("Model loaded successfully.")

    classNames = model.names
    print("Class names loaded successfully.")
    print(classNames)

    while True:
        print("Capturing frame...")
        success, img = cap.read()
        if not success:
            print("Failed to capture frames. Exiting...")
            break

        results = model(img, stream=True, conf=0.65)
        # Calculate FPS 
        # https://www.geeksforgeeks.org/python/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - old_frame_time)
        old_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)


        # Draw and label boxes on detected objects
        # https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                
                cv2.rectangle(img, (x1, y1), (x2, y2), (10, 241, 2), 3)

                
                if r.masks is not None:
                    for mask in r.masks.xy:
                        # The 'mask' variable is a list of (x, y) points
                        # We can draw it as a filled polygon
                        cv2.polylines(img, [mask.astype(int)], isClosed=True, color=(255, 0, 255), thickness=2)

                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 8)
                thickness = 2

                cv2.putText(img, f"{classNames[cls]}: {confidence}", org, font, fontScale, color, thickness)

        cv2.putText(img, fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (241, 2, 2), 2)
        #cv2.putText(img, str(results.count()), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (241, 2, 2), 2)
        cv2.imshow('BUS-APS', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()