import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchattacks
import torchvision
import timm
from timm.models import xcit, resnet, convnext
from timm.models.registry import register_model
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
import os
import pandas as pd
import warnings
from IPython.display import display
from PIL import Image
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((100, 100)),
    # transforms.RandomResizedCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    # TODO: CIFAR10, MNIST, FASHION MNIST Normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def resnet_pytorch(size=50, pretrained=False, **kwargs):
    assert size in [18, 34, 50, 101, 152], "Invalid ResNet size"

    n_classes = kwargs.get("num_classes", 1000)
    if "num_classes" in kwargs:
        kwargs.pop("num_classes")

    if size == 18:
        get_model = torchvision.models.resnet18
    elif size == 34:
        get_model = torchvision.models.resnet34
    elif size == 50:
        get_model = torchvision.models.resnet50
    elif size == 101:
        get_model = torchvision.models.resnet101
    else:
        get_model = torchvision.models.resnet152
    
    model = get_model(pretrained=pretrained, **kwargs)

    # Change the output layer
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    # Weight initialization of the output layer
    nn.init.xavier_normal_(model.fc.weight)
    return model

model = resnet_pytorch(size=18, pretrained=True)



files = glob.glob('./data/roam_dataset/**/*.jpg', recursive=True)
preds = []
for f in tqdm(files, total=len(files)):
    img = Image.open(f)
    img = transform(img)
    img = img.unsqueeze(0)
    # img = img.to(device)
    # print(img.shape)
    out = model(img)
    # print(out)
    label = torch.argmax(out).item()
    preds.append([f, label])

df_preds = pd.DataFrame(preds, columns=['file', 'label'])


fastercnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
fastercnn.eval()
print(fastercnn)


print(vars(fastercnn).keys())
print(dir(fastercnn))
print(fastercnn.backbone)

# get number of classes
n_classes = fastercnn.roi_heads.box_predictor.cls_score.out_features
print(n_classes)

random_file = np.random.choice(files)
img = Image.open(random_file)
img = transform(img)
img = img.unsqueeze(0)
out = fastercnn(img)
print(out)
