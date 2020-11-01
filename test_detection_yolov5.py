from detect import *
from PIL import Image, ImageDraw
import matplotlib.patches as patches
import matplotlib.pyplot as plt

image_path = '/home/aous/Desktop/Seaowl Stage/Ship classification/speed.jpeg'

def Test():
    print("Testing yolov5")
    assert(torch.cuda.is_available())
    img = Image.open(image_path)
    img_vect = img
    model = load_model()
    predictions = detect(model, img_vect)
    assert(len(predictions) == 1)

    img = np.array(img)
    _, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)
    ax.set_title('yolov5 detection')

    for (x1,y1,x2,y2, score, clas) in predictions:
        box_h = y2 -y1
        box_w = x2 - x1
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(bbox)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    Test()
