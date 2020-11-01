from Detect.yolov3.detect import *
from PIL import Image, ImageDraw
import matplotlib.patches as patches
import matplotlib.pyplot as plt

image_path = './Detect/test_image/military.jpeg'

def Test():
    #Testing yolov3
    print("Testing yolov3")
    assert(torch.cuda.is_available())
    img = Image.open(image_path)
    img_vect = img.convert('RGB')
    model = load_model()
    predictions = detect_image(model, img_vect)
    assert(len(predictions) == 1)

    img = np.array(img)
    _, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)
    ax.set_title('yolov3 detection')


    for (x1,y1,x2,y2, score, conf, clas) in predictions:

        box_h = y2 -y1
        box_w = x2 - x1
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(bbox)
    plt.axis('off')
    plt.show()
