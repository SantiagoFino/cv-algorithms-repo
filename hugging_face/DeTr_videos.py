import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from deep_sort_realtime.deepsort_tracker import deepsort_tracker


sample = 2
DATA_PATH = f'/content/drive/MyDrive/Documents/cv_projects/data/video_sample{sample}.mp4'
PREDICTIONS_PATH = f'/content/drive/MyDrive/Documents/cv_projects/predictions/video_predictions{sample}.avi'


class ObjectDetection:
    def __init__(self, capture, prediction_path):
        self.capture = cv2.VideoCapture(filename=capture)
        self.prediction_path = prediction_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model().to(self.device)
        self.processor = self.load_processor()
        self.writer = self.generate_writer()
        self.tracker = DeepSort(max_age=30,
                                n_init=2,
                                nms_max_overlap=1.0,
                                max_cosine_distance=0.3,
                                nn_budget=None,
                                override_track_class=None,
                                embedder="mobilenet",
                                half=True,
                                bgr=True,
                                embedder_gpu=True,
                                embedder_model_name=None,
                                embedder_wts=None,
                                polygon=False,
                                today=None)
        self.tracks = []

    def load_model(self):
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
        return model

    def load_processor(self):
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
        return processor

    def generate_writer(self):
        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(filename=self.prediction_path,
                                 fourcc=fourcc,
                                 fps=20,
                                 frameSize=(frame_width,  frame_height))
        return writer

    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs

    def draw_boxes(self, results, image):
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.tolist()
            # Draws the boxes of the predictions
            cv2.rectangle(img=image,
                        pt1=(int(box[0]), int(box[1])),
                        pt2=(int(box[2]), int(box[3])),
                        color=(0, 255, 0),
                        thickness=2)

            cv2.putText(img=image,
                        text=f"{self.model.config.id2label[label.item()]}: {round(score.item(), 3)}",
                        org=(int(box[0]), int(box[1])-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(255, 255, 255),
                        thickness=2)
        return image

    def detections_standarization(self, detections):
        # detections must be in the format ([left,top,right,bottom], confidence, detection_class)
        base_labels_list = detections['labels'].cpu().detach().numpy().tolist()

        boxes_list = detections['boxes'].cpu().detach().numpy().tolist()
        scores_list = detections['scores'].cpu().detach().numpy().tolist()
        labels_list = [self.model.config.id2label[label] for label in base_labels_list]

        combined_detections = []
        for box, score, label in zip(boxes_list, scores_list, labels_list):
            combined_detections.append([box, score, label])

        return combined_detections

    def track_detection(self, detections, image):
        # detections must be in the format ([left,top,right,bottom], confidence, detection_class)
        detections = self.detections_standarization(detections)
        # Updates the tracker
        self.tracks = self.tracker.update_tracks(raw_detections=detections,
                                                 frame=image)
        print("--------New Frame--------")
        for track in self.tracks:
            print(f"{track.get_det_class()}: {track.track_id}")


    def __call__(self):
        assert self.capture.isOpened(), "Cannot capture source"
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert the frame to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Computes the image predictions
            outputs = self.predict(pil_image)

            # Resizes the image
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs,
                                                                   target_sizes=target_sizes,
                                                                   threshold=0.9)[0]
            numpy_image = np.array(pil_image.convert('RGB'))
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            # Draws the bounding boxes on the frame
            numpy_image = self.draw_boxes(results=results,
                                          image=numpy_image)

            # Updates the tracker
            self.track_detection(detections=results,
                                 image=numpy_image)

            # add the frame to the video
            self.writer.write(numpy_image)

            # set a key to stop the video processing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        self.writer.release()
        cv2.destroyAllWindows()