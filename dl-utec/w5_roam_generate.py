"""
https://github.com/inspire-group/adv-patch-paper-list
https://github.com/SamSamhuns/yolov5_adversarial
https://github.com/jhayes14/adversarial-patch/blob/master/make_patch.py
https://github.com/search?q=adversarial%20patch&type=repositories

IDEAS:
    - YOLO detecta carros y personas
        - Crear un parche adversarial que haga que YOLO no detecte carros
        - Crear un parche adversarial que haga que YOLO no detecte personas
    - LSFM detecta personas
        - Crear un parche adversarial que haga que LSFM no detecte personas
    - Faster R-CNN detecta personas

    - Cuando haga el adversarial patch generation ...
        - False positives
            - Tengo que hacer que detecte algo que no es
        - False negatives
PAPERS:
    CityPersons: A Diverse Dataset for Pedestrian Detection
        https://arxiv.org/pdf/1702.05693
    LSFM:
        https://openaccess.thecvf.com/content/CVPR2023/papers/Khan_Localized_Semantic_Feature_Mixers_for_Efficient_Pedestrian_Detection_in_Autonomous_CVPR_2023_paper.pdf
    FasterRCNN:
        https://arxiv.org/pdf/1506.01497
    Enhancing Object Detection for Autonomous Driving by Optimizing Anchor Generation and Addressing Class Imbalance
        https://arxiv.org/pdf/2104.03888
    Detectron2:
        https://github.com/facebookresearch/detectron2
    CascadeRCNN:
        https://arxiv.org/pdf/1712.00726
    Patches:
        https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf
        https://arxiv.org/pdf/2005.08087
DATASETS:
    KITTI
    Waymo
    nuScenes
        https://arxiv.org/pdf/1903.11027
    Cityscapes
    Kaggle
        https://www.kaggle.com/datasets/alincijov/self-driving-cars/code
BLOGS:
    https://artificio.org/blog/why-has-autonomous-driving-failed-perspectives-from-peru-and-insights-from-neuroai
CHATGPT:
    Parche para falso positivo: 
        Se coloca en una imagen sin el objeto objetivo para que el modelo lo detecte erróneamente. El parche induce una detección de algo que no está presente.

    Parche para falso negativo: 
        Se coloca sobre o cerca del objeto real para que el modelo no lo detecte. El parche enmascara el objeto y lo oculta del modelo.
"""


import torch
import torch.nn as nn
import supervision as sv
import torch.nn.functional as F
from torchinfo import summary
from ultralytics import YOLO
import torchattacks
import cv2
import torchvision
import timm
from tqdm import tqdm
from timm.models import xcit, resnet, convnext
from timm.models.registry import register_model
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import glob
import os
import pandas as pd
import warnings
from IPython.display import display
import math
import random
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# PyTorch ResNet
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

    # # Freeze all layers except the output layer
    # for name, param in model.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False
    
    return model


# -----------------------------------------------------------------------------
# YOLO
# -----------------------------------------------------------------------------

model = YOLO("yolov8x-seg.pt")

print(f'n_classes: {len(model.names)}')
# 0: person | 1: bicycle | 2: car | 3: motorcycle | 5: bus | 6: train | 7: truck
print(json.dumps(model.names, indent=2))

print(model)

# get the YOLO classifier
print(vars(model).keys())


# -----------------------------------------------------------------------------
# Faster R-CNN
# -----------------------------------------------------------------------------

fastercnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
print(fastercnn)


print(vars(fastercnn).keys())
print(dir(fastercnn))
print(fastercnn.backbone)

# get number of classes
n_classes = fastercnn.roi_heads.box_predictor.cls_score.out_features
print(n_classes)

"""
https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://github.com/shashankag14/fine-tune-object-detector
https://learnopencv.com/fine-tuning-faster-r-cnn/
https://github.com/pytorch/tutorials/blob/main/intermediate_source/torchvision_tutorial.py
"""


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

labels = { 1:'car', 2:'truck', 3:'person', 4:'bicycle', 5:'traffic light'}

df = pd.read_csv('./data/kaggle-self-driving/labels_train.csv')
df = (
    df.loc[df['class_id'].isin([1, 3])]
)
classes = df['class_id'].unique()

for i in classes:
    idx_random = random.choice(df[df['class_id'] == i].index)    
    row = df.loc[idx_random]
    img_name, xmin, xmax, ymin, ymax, class_id = row.values
    img_path = f'./data/kaggle-self-driving/images/{img_name}'
    img = plt.imread(img_path)

    results = model(img_path, conf=0.5, iou=0.5)[0]

    plt.figure(figsize=(8, 10))
    plt.title("Label " + labels[i])
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='yellow', fill=False, linewidth=2))
    
    plt.show()

    plot = results.plot()
    plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    plt.imshow(plot)
    
    detections = sv.Detections.from_ultralytics(results)
    print(detections.class_id)

print(vars(detections).keys())
print(f"{detections.xyxy.shape=} {detections.xyxy=}")
print(f"{detections.confidence.shape=} {detections.confidence=}")
print(f"{detections.mask.shape=} {detections.mask=}")
print(f"{detections.class_id.shape=} {detections.class_id=}")
print(f"{detections.data=}")

# plot the mask
merged_mask = detections.mask.sum(axis=0)
merged_mask = (merged_mask * 255).astype('uint8')
plt.imshow(merged_mask)

# get the dimensions of each object
for tup in detections.xyxy:
    print(f"{tup=}")
    x1, y1, x2, y2 = tup
    w, h = x2 - x1, y2 - y1
    print(f"{w=} {h=}")

# distribution of the detections
data = []
print(df['class_id'].value_counts())
df_sample = df.groupby('class_id').apply(lambda x: x.sample(250, random_state=seed)).reset_index(drop=True)

for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    img_name, xmin, xmax, ymin, ymax, class_id = row.values
    img_path = f'./data/kaggle-self-driving/images/{img_name}'
    img = plt.imread(img_path)

    results = model(img_path, conf=0.5, iou=0.5, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # get the dimensions of each object
    dimensions = [(x2 - x1, y2 - y1, [x1, y1, x2, y2]) for x1, y1, x2, y2 in detections.xyxy]
    data.extend([img_path, c_id, w, h, bbox] for c_id, (w, h, bbox) in zip(detections.class_id, dimensions))
print(f"{len(data)=}")

df_data = (
    pd.DataFrame(data, columns=['im_path', 'class_id', 'width', 'height', 'bbox'])
    .loc[lambda df: df['class_id'].isin([0, 1, 2, 3, 5, 6])]
)
df_data['resolution'] = df_data['width'] * df_data['height']
threshold = 50
df_data['is_valid'] = df_data.apply(lambda x: x['width'] >= threshold and x['height'] >= threshold, axis=1)

df_valid = df_data.loc[df_data['is_valid'] == True]

print(df_valid['resolution'].describe())
print(df_valid['class_id'].value_counts())
print(df_valid['is_valid'].value_counts())

for idx, row in df_valid.loc[df_valid['is_valid'] == True].iterrows():
    img_path, class_id, width, height, bbox, _, _ = row.values
    os.makedirs(f'./data/roam_dataset/{class_id}_{model.names[class_id]}', exist_ok=True)
    img = plt.imread(img_path)
    x1, y1, x2, y2 = bbox
    
    patch = img[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)]
    # plt.imshow(patch)
    # plt.imshow(img)
    # plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='yellow', fill=False, linewidth=2))
    # plt.show()

    # save the patch
    patch = Image.fromarray(patch)
    patch.save(f'./data/roam_dataset/{class_id}_{model.names[class_id]}/{os.path.basename(img_path).replace(".jpg", "")}_{idx}.jpg')


# -----------------------------------------------------------------------------
# Adversarial Patch Generation
# -----------------------------------------------------------------------------

print(f"{len(df_valid['im_path'].unique())=}", f"{len(df_valid)=}")

patch_size = (50, 50, 3)
patch = np.random.uniform(0, 1, patch_size)
plt.imshow(patch)

patch = torch.tensor(patch, requires_grad=True)

def adversarial_loss(past_confidence, new_confidence, past_class_ids, new_class_ids, target_class_id):
    """
    Calcula la pérdida adversaria:
    - Minimizar la confianza en la clase objetivo (falso negativo).
    - Si la clase cambia completamente (falso positivo), la pérdida será mínima (0).
    """
    past_conf = torch.tensor(past_confidence, dtype=torch.float32, requires_grad=True)
    new_conf = torch.tensor(new_confidence, dtype=torch.float32, requires_grad=True)

    # Creamos máscaras para identificar las detecciones de la clase objetivo
    past_target_mask = torch.tensor(past_class_ids == target_class_id, dtype=torch.bool)
    new_target_mask = torch.tensor(new_class_ids == target_class_id, dtype=torch.bool)

    # Extraemos las confidencias solo de la clase objetivo antes y después del parche
    target_past_conf = past_conf[past_target_mask]
    target_new_conf = new_conf[new_target_mask]

    # Si había detecciones de la clase objetivo pero ya no las hay, consideramos un ataque exitoso (pérdida = 0)
    if len(target_past_conf) > 0 and len(target_new_conf) == 0:
        # Si el ataque causó la desaparición de la clase objetivo (falso negativo)
        loss = torch.tensor(0.0)  # No hay confianza restante en la clase objetivo
    elif len(target_new_conf) > 0:
        # Si aún hay detecciones de la clase objetivo, minimizar su confianza
        loss = torch.mean(target_new_conf)
    else:
        # Si no había detecciones de la clase objetivo en primer lugar, la pérdida también es 0
        loss = torch.tensor(0.0)

    return loss


# -----------------------------------------------------------------------------
# Maximizing False Negatives in Car Detections (Class ID: 2)
# -----------------------------------------------------------------------------

target_class_id = 2
num_steps = 1

target_df = df_valid.loc[df_valid['class_id'] == target_class_id]['im_path']
for step in range(num_steps):
    total_loss = 0.0
    plt.imshow(patch.detach().numpy())
    for image in tqdm(target_df, total=len(target_df)):
        # forward pass
        results = model(img_path, conf=0.5, iou=0.5)[0]
        detections = sv.Detections.from_ultralytics(results)
        # label = torch.argmax(out).item()
        past_ids, past_confidence = detections.class_id, detections.confidence
        
        os.makedirs('./data/roam_dataset/patched', exist_ok=True)

        patch.requires_grad = True
        
        # create image with patches in the car detections
        def create_patched_image(image, detections, patch):
            """TODO: Locate the patch in the car detections"""
            # save image
            img = Image.fromarray(img)
            path = f'./data/roam_dataset/patched/patched_{os.path.basename(image)}'
            img.save(path)

            return img, path
        
        adv_img, path = create_patched_image(image, detections, patch)
        plt.imshow(adv_img)
        detections = sv.Detections.from_ultralytics(model(adv_img, conf=0.5, iou=0.5)[0])

        break
    break
        
    #     patched_image = create_patched_image(image, detections, patch)
    #     detections = sv.Detections.from_ultralytics(model(patched_image, conf=0.5, iou=0.5)[0])
    #     new_ids, new_confidence = detections.class_id, detections.confidence
        
    #     # Calcular la pérdida adversaria (maximizar el falso negativo)
    #     loss = adversarial_loss(past_confidence, new_confidence, past_ids, new_ids, target_class_id)
    #     # print(f'Loss: {loss.item()}')
    #     if loss.item() > 0:
    #         optimizer.zero_grad()
    #         loss.backward()
        
    #     total_loss += loss.item()
    
    # print(f'Step {step + 1}/{num_steps}, Loss: {total_loss/len(target_df)}')
