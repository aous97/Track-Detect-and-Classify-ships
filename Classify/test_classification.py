from Classify.config import *
from Classify.produce_classification import *
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])
image_path = './Classify/test_image/cargo.jpeg'

def Test():
    assert(torch.cuda.is_available())
    img_vect = Image.open(image_path).convert('RGB').resize((256,256))
    for model_name in model_dict:
        print("Testing "+ model_name +"\n")
        model = load_model(model_name)
        pred = get_pred_vect(model, img_vect)
        labels = get_label(pred)
        if labels[0] != 1:
            print("Error: Wrong Classification \n")
        else:
            if labels[1] != 'Bulk Carrier':
                print("Error: Wrong label Name \n")
            else:
                print("Sucess prediction " + model_name +"\n")
