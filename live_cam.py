import cv2
import torch
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from transformers import YolosImageProcessor, YolosForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from torchvision import transforms
import torch.nn.functional as F
import io

import utils

'''

'''



device = torch.device("cuda")

yolo_processor = YolosImageProcessor.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')
yolo_model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection').to(device)

trocr_processor = TrOCRProcessor.from_pretrained('GudaGuda05/Trocr_License_Plate', use_fast=False)
trocr_model = VisionEncoderDecoderModel.from_pretrained('GudaGuda05/Trocr_License_Plate').to(device)

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Capture the frame
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame).to(device)
  
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Detect LP's from frame
    pixel_values = yolo_processor(frame_tensor, return_tensors='pt').pixel_values.to(device)
    boxes = yolo_processor.post_process_object_detection(yolo_model(pixel_values), threshold=.9)[0]["boxes"]
    
    if boxes.size()[0] > 0:
        boxes = utils.format_boxes(boxes, frame)
        images = utils.crop_objects(boxes, frame,device)
        if images != None:
            pixel_values = trocr_processor(images, return_tensors='pt').pixel_values.to(device)
            outputs = trocr_model.generate(pixel_values)
            labels = trocr_processor.post_process_image_text_to_text(outputs)

        frame = utils.draw_rectangle(frame, boxes, labels)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break


camera.release()
cv2.destroyAllWindows()