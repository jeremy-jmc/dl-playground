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
    transforms.Resize((32, 32)),
    # transforms.RandomResizedCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    # TODO: CIFAR10, MNIST, FASHION MNIST Normalization
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# -----------------------------------------------------------------------------
# Create model
# -----------------------------------------------------------------------------


# PyTorch ResNet-50
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


model = resnet_pytorch(size=18, pretrained=True, num_classes=30)


# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------

model.load_state_dict(torch.load('model_resnest18_1epoch.pth', map_location=device))


# -----------------------------------------------------------------------------
# Generate submission
# -----------------------------------------------------------------------------

def generate_submission(model, nivel, path_test_set = './data/NIVEL1/NIVEL1/TEST1_3/**/*.png', tags=''):
    submission_set = glob.glob(path_test_set, recursive=True)
    submission_set = pd.DataFrame(submission_set, columns=['Path'])

    print(len(submission_set))

    # Inference
    model.eval()
    model.to(device)
    preds = []

    for i, row in tqdm(submission_set.iterrows(), total=len(submission_set)):
        img_path = row['Path']
        image = plt.imread(img_path)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        image = transform(image).unsqueeze(0).to(device)
        # print(image.shape)
        pred = model(image)
        pred = torch.argmax(pred, dim=1).item()
        preds.append(pred)

    assert len(preds) == len(submission_set)

    submission_set['Label'] = preds
    submission_set['ImageId'] = submission_set['Path'].apply(lambda x: x.split('\\')[-1])

    print(len(submission_set))
    display(submission_set)

    submission_set[['ImageId', 'Label']].to_csv(f'submission_{nivel}_{tags}.csv', index=False)


tags = 'final result'
generate_submission(model, 'test1', './data/NIVEL1/NIVEL1/TEST1_3/**/*.png', tags=tags)
generate_submission(model, 'test2', './data/NIVEL2/NIVEL2/TEST2_3/**/*.png', tags=tags)
generate_submission(model, 'test3', './data/NIVEL3/NIVEL3/TEST3_3/**/*.png', tags=tags)





