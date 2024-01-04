from utils import *
from train import *
import os
import urllib.request
from urllib.error import HTTPError
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import tabulate
import numpy as np
from IPython.display import HTML, display

# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial5/"
# Files to download
pretrained_files = [
    "GoogleNet.ckpt",
    "ResNet.ckpt",
    "ResNetPreAct.ckpt",
    "DenseNet.ckpt",
    "tensorboards/GoogleNet/events.out.tfevents.googlenet",
    "tensorboards/ResNet/events.out.tfevents.resnet",
    "tensorboards/ResNetPreAct/events.out.tfevents.resnetpreact",
    "tensorboards/DenseNet/events.out.tfevents.densenet",
]
# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                e,
            )


googlenet_model, googlenet_results = train_model(
    model_name="GoogleNet",
    model_hparams={"num_classes": 10, "act_fn_name": "relu"},
    optimizer_name="Adam",
    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
)
print("GoogleNet Results", googlenet_results)

resnet_model, resnet_results = train_model(
    model_name="ResNet",
    model_hparams={"num_classes": 10, "c_hidden": [16, 32, 64], "num_blocks": [3, 3, 3], "act_fn_name": "relu"},
    optimizer_name="SGD",
    optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
)
print("ResNet Results", resnet_results)

resnetpreact_model, resnetpreact_results = train_model(
    model_name="ResNet",
    model_hparams={
        "num_classes": 10,
        "c_hidden": [16, 32, 64],
        "num_blocks": [3, 3, 3],
        "act_fn_name": "relu",
        "block_name": "PreActResNetBlock",
    },
    optimizer_name="SGD",
    optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
    save_name="ResNetPreAct",
)
print("ResNetPreAct Results", resnetpreact_results)

densenet_model, densenet_results = train_model(
    model_name="DenseNet",
    model_hparams={
        "num_classes": 10,
        "num_layers": [6, 6, 6, 6],
        "bn_size": 2,
        "growth_rate": 16,
        "act_fn_name": "relu",
    },
    optimizer_name="Adam",
    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
)
torch.cuda.empty_cache()


all_models = [
    ("GoogleNet", googlenet_results, googlenet_model),
    ("ResNet", resnet_results, resnet_model),
    ("ResNetPreAct", resnetpreact_results, resnetpreact_model),
    ("DenseNet", densenet_results, densenet_model),
]
table = [
    [
        model_name,
        f"{100.0*model_results['val']:4.2f}%",
        f"{100.0*model_results['test']:4.2f}%",
        f"{sum(np.prod(p.shape) for p in model.parameters()):,}",
    ]
    for model_name, model_results, model in all_models
]
display(
    HTML(
        tabulate.tabulate(table, tablefmt="html", headers=["Model", "Val Accuracy", "Test Accuracy", "Num Parameters"])
    )
)

print(tabulate.tabulate(table, headers=["Model", "Val Accuracy", "Test Accuracy", "Num Parameters"]))