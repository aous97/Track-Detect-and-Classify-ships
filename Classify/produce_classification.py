from Classify.config import *
import time
from torch.autograd import Variable

def classifier(out_size, num_classes):
    classifier = nn.Sequential(nn.Linear(out_size, int(out_size/2)), nn.BatchNorm1d(int(out_size/2)) , nn.ReLU(),
                               nn.Dropout(p=0.1), nn.Linear(int(out_size/2), num_classes))
    return(classifier)
def googlenet(num_classes):
    net = models.googlenet(pretrained=True, progress=True)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    net.aux1.fc2 = classifier(net.aux1.fc2.in_features, num_classes)
    net.aux2.fc2 = classifier(net.aux2.fc2.in_features, num_classes)
    return(net)

def Resnet34(num_classes):
    net = models.resnet34(pretrained=True, progress=True)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)

def Resnet50(num_classes):
    net = models.resnet50(pretrained=True, progress=True)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)

def Resnet152(num_classes):
    net = models.resnet152(pretrained=True, progress=True)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)

model_dict = {
    'resnet34': [Resnet34, './Classify/models/r_net81%_23.pth'],
    'resnet152': [Resnet152, './Classify/models/Res152_87%.pth'],
    'resnet50': [Resnet50, './Classify/models/Res50_84%.pth'],
}

def load_model(model_name, weights_path = None, num_classes = 23, cuda = True):
    print("Loading classification model ...")
    model_class = model_dict[model_name][0]
    model = model_class(num_classes)
    if weights_path != None:
        weight = torch.load(weights_path)
    else:
        weight = torch.load(model_dict[model_name][1])
    model.load_state_dict(weight)
    model.eval()
    if cuda:
        model = model.cuda()
    return(model)

def get_label(vect):
    if len(vect) == 23:
        names = names23
    elif len(vect) == 26:
        names = names26
    else:
        print("Invalid class number for classification")
        assert(False)
    
    m = 0
    for i in range(len(vect)):
        if vect[i] > vect[m]:
            m = i
    return(m, names[m])

def get_pred_vect(model, img):
    image_obj = transform(img).unsqueeze(0)
    image_obj = Variable(image_obj.type(torch.cuda.FloatTensor))
    pred = model(image_obj)
    Softner = nn.Softmax(dim = 0)
    return(Softner(pred[0]).tolist())
