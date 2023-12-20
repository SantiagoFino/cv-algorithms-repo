from typing import Any
import torch
import numpy as np
import cv2
import time
from ultralytics import RTDETR
import supervision as sv


class DETRClass:

    def __init__(self, capture_index) -> None:
        """
        Constructor
        """
        self.capture_index = capture_index
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = RTDETR("rtdetr-l.pt")
        self.CLASS_NAMES_DICT = self.model.model.names

        print("classes: ", self.CLASS_NAMES_DICT)

        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(),
                                             thickness=3,
                                             text_thickness=3,
                                             text_scale=1.5)
        
    
    def plot_bboxes(self, results, frame):
        """
        """
        # Extract the results
        boxes = results[0].boxes.cpu().numpy()
        class_id = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy

        class_id = class_id.as_type(np.int32)


        detections = sv.Detections(xyxy=xyxy,
                                   class_id=class_id,
                                   confidence=conf)
        
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}" 
                       for xyxy, mask, confidence, class_id, track_id in detections]
        
        frame_result = self.box_annotator(frame, detections, self.labels)

        return frame_result
    

    def __call__(self):
        
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()

            results = self.model.predict(frame)
            frame = self.plot_bboxes(results=results,
                                     frame=frame)
            end_time = time.perf_counter()

            fps = 1 / (end_time - start_time)
            cv2.putText(img=frame, 
                        text=f"FPS: {fps:.2f}", 
                        org=(20, 70),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2)
            cv2.imshow("DETR", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

transformer_detector = DETRClass(capture_index=0)
transformer_detector()
