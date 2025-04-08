import cv2
import torch
from PIL import ImageOps, Image
import random

'''
Util file thats used mainly for live_cam.py
'''

def crop_objects(boxes, image):
    #padding_to = 400
    images = []
    for box in boxes:

        x_min, y_min, x_max, y_max = box.tolist()

        # Crop image
        try:
            img_cropped = image.crop([x_min, y_min, x_max, y_max])
            angle = random.uniform(-30,30)
            images.append(ImageOps.grayscale(img_cropped).convert("RGB"))
        except:
            print(x_min, y_min, x_max, y_max)
            print(box)
            image.show()
            exit()

    return images

def format_boxes(boxes, image):
    new_boxes = []
    for box in boxes:
        normalized = len(box[box < 1]) == 4
        
        x_min, y_min, x_max, y_max = box

        if normalized:
            width, height = image.size
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
    return torch.tensor(new_boxes)

def draw_rectangle(image,boxes,labels):
    for i,box in enumerate(boxes):
        try:
            label = labels[i]
        except:
            label = None

        x_min, y_min, x_max, y_max = box.tolist()
        image = cv2.rectangle(image, (x_min,y_max), (x_max,y_min), color=(0,0,255), thickness=2)
        image = cv2.putText(image, label, (x_min,y_max+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image
