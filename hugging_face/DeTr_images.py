import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import cv2
from transformers import DetrForObjectDetection, DetrImageProcessor



processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

# image processing
image_path = 'img_autonorte.jpeg'
image = Image.open("data/" + image_path)

# Converts the image into a numpy.array 
numpy_image = np.array(image.convert('RGB'))
numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 

# reprocess the image according to the used model
inputs = processor(images=image, return_tensors="pt")

# run the model over the image
outputs = model(**inputs)

# resizes the image for the post_process_object_detection
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, 
                                                  target_sizes=target_sizes, 
                                                  threshold=0.2)[0]
                      
# iterates over the results in order to draw bounding boxes and labels on the frame
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = box.tolist()
    # Draws the boxes of the predictions
    cv2.rectangle(img=numpy_image, 
                  pt1=(int(box[0]), int(box[1])), 
                  pt2=(int(box[2]), int(box[3])), 
                  color=(0, 255, 0), 
                  thickness=2)
    
    cv2.putText(img=numpy_image, 
                text=f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", 
                org=(int(box[0]), int(box[1])-10), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(255, 255, 255), 
                thickness=2)

save_dir = "predictions/hugging_face/"
cv2.imwrite(os.path.join(save_dir, "prediction_results.jpg"), numpy_image)

labels = [model.config.id2label[label] for label in results["labels"].tolist()]

df = pd.DataFrame({"object": labels, "score": list(results["scores"].tolist())})
df.to_json("predictions/hugging_face/results_json.json", indent=4)
