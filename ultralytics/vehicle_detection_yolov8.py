from ultralytics import YOLO
from PIL import Image


model = YOLO(model="yolov8n.pt")
model.info()

source = "https://www.eltiempo.com/files/image_950_534/uploads/2022/03/04/6222e496a9fa1.jpeg"

results = model.predict(source=source, conf=0.1)

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()
    im.save('predictions/ultralytics/yolov8_result.jpg')

