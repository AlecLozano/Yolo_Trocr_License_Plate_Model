import cv2
import torch

'''
Util file thats used mainly for live_cam.py
'''

def crop_objects(boxes, image, device):
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
