import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from deep_sort_realtime.deepsort_tracker import DeepSort


sample = 2

DATA_PATH = f'/content/drive/MyDrive/Documents/cv_projects/data/video_sample{sample}.mp4'
PREDICTIONS_PATH = f'/content/drive/MyDrive/Documents/cv_projects/predictions/video_predictions{sample}.avi'
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b


class DeTrDetector:
    def __init__(self, capture, prediction_path, classes=CLASSES):
        """
        Constructor
        Param:
            capture:
            prediction_path:
        """
        self.capture = cv2.VideoCapture(capture)
        self.prediction_path = prediction_path
        self.classes = classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.post_processor = self.load_model()
        self.processor = self.load_processor()
        self.writer = self.generate_writer()
        self.tracker = self.generate_tracker()
        self.tracks = []

    def generate_tracker(self):
        pass

    def load_model(self):
        model, post_processor = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50',
                                               pretrained=True,
                                               return_postprocessor=True)
        return model.to(self.device), post_processor

    def load_processor(self):
        processor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(800),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return processor

    def generate_tracker(self):
        tracker = DeepSort(max_age=30,
                                n_init=2,
                                nms_max_overlap=3.0,
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
        return tracker

    def generate_writer(self):
        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(filename=self.prediction_path,
                                 fourcc=fourcc,
                                 fps=fps,
                                 frameSize=(frame_width,  frame_height))
        return writer

    def predict(self, image, threshole=0.9):
        # Image preprocessing
        input = self.processor(image).unsqueeze(0).to(self.device)

        # pass the input image for the model
        outputs = self.model(input)

        # get the predictions that scores a value grater than 0.9
        scores = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = scores.max(-1).values > threshole

        # prepare the output
        results = {}
        results["boxes"] = rescale_bboxes(outputs['pred_boxes'][0, keep],
                                          image.size,
                                          device=self.device)
        results["scores"] = scores.max(-1).values[keep]
        results["labels"] = [self.classes[score.argmax()] for score in scores[keep]]
        return results

    def draw_boxes(self, results, image):
        for score, box, label in zip(results["scores"], results["boxes"], results["labels"]):
            box = box.tolist()
            # Draws the boxes of the predictions
            cv2.rectangle(img=image,
                        pt1=(int(box[0]), int(box[1])),
                        pt2=(int(box[2]), int(box[3])),
                        color=(0, 255, 0),
                        thickness=2)

            cv2.putText(img=image,
                        text=f"{label}: {score:0.2f}",
                        org=(int(box[0]), int(box[1])-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(255, 255, 255),
                        thickness=2)
        return image

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
            results = self.predict(pil_image)

            # Cast the image as a numpy image
            numpy_image = np.array(pil_image.convert('RGB'))
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            # Draws the bounding boxes on the frame
            numpy_image = self.draw_boxes(results=results,
                                          image=numpy_image)

            # add the frame to the video
            self.writer.write(numpy_image)

            # set a key to stop the video processing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        self.writer.release()
        cv2.destroyAllWindows()