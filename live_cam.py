import cv2
import torch
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from transformers import YolosImageProcessor, YolosForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from torchvision import transforms
import torch.nn.functional as F
import io



def crop_objects(boxes, image):
    #padding_to = 400
    images = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        # Crop image
        try:
            images.append(torch.tensor(image[x_min:x_max, y_min:y_max]).to(device))
        except:
            print("box", [x_min, y_min, x_max, y_max], end='\n\n\n')
            print(image)
            return None

    return images

def format_boxes(boxes, image):
    new_boxes = []
    for box in boxes:
        normalized = all(coord <= 1 for coord in box)
            
        x_min, y_min, x_max, y_max = box

        if normalized:
            height, width, _ = image.shape
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)

        # Ensure valid coordinates
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        new_boxes.append([x_min, y_min, x_max, y_max])
    return new_boxes

def draw_rectangle(image,boxes,labels):
    for i,box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        image = cv2.rectangle(image, (x_min,y_max), (x_max,y_min), color=(0,0,255), thickness=2)
        image = cv2.putText(image, labels[i], (x_min,y_max+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image

device = torch.device("cuda")

yolo_processor = YolosImageProcessor.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')
yolo_model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection').to(device)

trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed', use_fast=False)
trocr_model = VisionEncoderDecoderModel.from_pretrained(r'Yolo\model\model.pt').to(device)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Capture the frame
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame).to(device)
  
    # If frame is read correctly, ret will be True
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the frame
    pixel_values = yolo_processor(frame_tensor, return_tensors='pt').pixel_values.to(device)
    boxes = yolo_processor.post_process_object_detection(yolo_model(pixel_values), threshold=.9)[0]["boxes"]
    
    if boxes.size()[0] > 0:
        boxes = format_boxes(boxes, frame)
        images = crop_objects(boxes, frame)
        if images != None:
            pixel_values = trocr_processor(images, return_tensors='pt').pixel_values.to(device)
            outputs = trocr_model.generate(pixel_values)
            labels = trocr_processor.post_process_image_text_to_text(outputs)

        frame = draw_rectangle(frame, boxes, labels)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break


# Release the camera and close all windows
camera.release()
