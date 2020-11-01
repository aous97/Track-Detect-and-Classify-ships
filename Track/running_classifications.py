from Classify.produce_classification import *
from PIL import Image
import time
import cv2
from torchvision.ops import roi_pool
import torchvision.transforms as transforms

class Classifications(object):
    def __init__(self, classifier, num_classes = 23, eps = 1e-7):
        self.num_classes = num_classes
        self.tracked_classes = {}
        self.classifier = classifier
        self.eps = eps

        def crop_resize(image_obj, coords, x_size = 256, y_size = 256):
            coords = (coords[0], coords[1], coords[2], coords[3])
            cropped_image = image_obj.crop(coords)
            resized_image = cropped_image.resize((x_size, y_size), Image.ANTIALIAS)
            return(resized_image.convert('RGB'))
        self.crop_resize = crop_resize

    def predict(self, tracks, img):
        Softner = nn.Softmax()
        #st = time.time()
        #transform(img)
        transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ])
        batch = transform(self.crop_resize(img, tracks[0][:4])).unsqueeze(0)
        #print(time.time() - st)
        for i in range(len(tracks) - 1):
            batch = torch.cat((batch, transform(self.crop_resize(img, tracks[i + 1][:4])).unsqueeze(0)), 0)
        #print(list((torch.Tensor(tracks[i][:4]) for i in range(len(tracks)))))
        #img = trans(img)
        #img /= 255.0
        #img = transform(img)
        #print(trans(img))
        #print(list((torch.Tensor(tracks[i][:4]).unsqueeze(0) for i in range(len(tracks)))))
        #batch = roi_pool(img.unsqueeze(0), list((torch.Tensor(tracks[i][:4]).unsqueeze(0) for i in range(len(tracks)))), (256, 256))
        #tr = transforms.ToPILImage()
        #print(torch.max(batch))
        #tr(batch[0]).show()
        #ok
        batch = Variable(batch.type(torch.cuda.FloatTensor))
        preds = self.classifier(batch)
        predictions = []
        for i in range(preds.shape[0]):
            predictions.append(Softner(preds[i]).tolist())
        #print(len(predictions), len(tracks))
        return(predictions)

    def update(self, tracks, img):
        if tracks.shape[0] == 0:
            return(None)
        n = self.num_classes
        ro = [0]*n
        eps = self.eps
        track_nums = tracks[:, 4]
        predictions = self.predict(tracks, img)
        deleted = []
        for box in self.tracked_classes:
            found = None
            idx = 0
            for i, track_num in enumerate(track_nums):
                if track_num == box:
                    found = tracks[i]
                    idx = i
                    break
            if found is not None:
                f = self.tracked_classes[box]
                p = predictions[idx]

                for i in range(n):
                    s = 0
                    for j in range(n):
                        s += p[j]*f[j]
                    ro[i] = s / (p[i]*f[i] + eps)
                for i in range(n):
                    f[i] = 1 / (ro[i] + eps)
                self.tracked_classes[box] = f

            else:
                deleted.append(box)
        for b in deleted:
            del self.tracked_classes[b]
        
        for i, track_num in enumerate(track_nums):
            new = not (track_num in self.tracked_classes)
            if new:
                self.tracked_classes[track_num] = predictions[i]
    
    def label_of_box(self, box):
        return(get_label(self.tracked_classes[box]))

        

