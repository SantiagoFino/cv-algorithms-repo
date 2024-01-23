import numpy as np
import torch
import cv2
from PIL import Image



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
    def __init__(self, model, processor, capture, prediction_path, classes):
        """
        Constructor
        Param:
            model:
            processor:
            capture:
            prediction_path:
            classes:
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = classes
        self.model = model.to(self.device)
        self.processor = processor.to(self.device)

        self.capture = cv2.VideoCapture(capture)
        self.prediction_path = prediction_path
        self.writer = self.generate_writer()

    def generate_writer(self):
        """
        Generates a video writter cv object. Fixes the frame width, height and the
        FPS of the outcome video
        Return:
            cv2.VideoWritter object
        """
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
        """
        Detects the objects that are in a input image and select the ones with an accuracy
        higher than the one specified in the parameters adding them into a result dictionary 
        that stores their bounding boxes, the scores and the labels.
        Params:
            image (): Image that is going to be process
            threshole (float): Threshole of the detection
        Return:
            dict with the predictions. The keys are "boxes" which value is a list with the 
            bounding boxes found at the input image, "scores" which value is a list with the
            accuracy of the found objects and "labels" which value is a list with the labels 
            of the predicted objects
        """
        input = self.processor(image).unsqueeze(0).to(self.device)
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
        """
        Given a dictionary with the results and its respective image, draw the bounding boxes making
        emphasis in the object labels and in their accuracy.
        Param:
            results (dict): dict with the predictions. The keys are "boxes" which value is a list with the 
            bounding boxes found at the input image, "scores" which value is a list with the
            accuracy of the found objects and "labels" which value is a list with the labels 
            of the predicted objects
            image (): Image related to the results dictionary
        """
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
        """
        Executes the object detection pipeline on a video stream. 

        The method captures frames from the video source, processes each frame to perform object detection, 
        draws bounding boxes around detected objects, and writes the processed frames to an output video. 
        The process continues until the video stream ends or a 'q' key is pressed.

        Steps:
        1. Asserts that the video capture is successfully opened.
        2. Reads frames from the video capture in a loop.
        3. Converts frames to PIL format for processing.
        4. Computes predictions for the current frame.
        5. Converts the PIL image back to a NumPy array and adjusts color channels.
        6. Draws bounding boxes on the frame based on the predictions.
        7. Writes the processed frame to the video writer.
        """
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