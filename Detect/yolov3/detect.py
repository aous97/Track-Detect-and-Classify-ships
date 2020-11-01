import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from Detect.yolov3.models import *
from Detect.yolov3.utils import *
#from Detect.yolov3.sort import *

#YOLOv3
config_path='./Detect/yolov3/config/yolov3.cfg'
weights_path='./Detect/yolov3/config/yolov3.weights'
class_path='./Detect/yolov3/config/coco.names'
img_size=416
conf_thres=0.5
nms_thres=0.4

def detect_image(model, img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(torch.cuda.FloatTensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)

    img = np.array(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    
    if detections[0] == None:
        return(None)
    for i, (x1,y1,x2,y2, score, conf, clas) in enumerate(detections[0]):
        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
        detections[0][i][0] = x1
        detections[0][i][1] = y1
        detections[0][i][2] = x1 + box_w
        detections[0][i][3] = y1 + box_h

    return detections[0]

def load_model():
    # Load model and weights
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    model.cuda()
    model.eval()
    return(model)


