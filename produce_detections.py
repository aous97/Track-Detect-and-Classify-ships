from Detect.yolov3.detect import load_model as yolov3_load
from detect import load_model as yolov5_load
from Detect.yolov3.detect import detect_image as yolov3detect
from detect import detect as yolov5detect

detect_model_dict = {
    'yolov3' : [yolov3_load, yolov3detect],
    'yolov5' : [yolov5_load, yolov5detect],
}
def get_model(model_name):
    print("loading detection model...")
    return(detect_model_dict[model_name][0]())

def get_detections(model_name, model, img):
    return(detect_model_dict[model_name][1](model, img))
