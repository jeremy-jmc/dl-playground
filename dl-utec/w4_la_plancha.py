# !python -m pip install timm torchinfo tqdm
# !python -m pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
"""
* https://github.com/dedeswim/vits-robustness-torch/tree/master
https://timm.fast.ai/
https://github.com/RobustBench/robustbench.github.io
https://robustbench.github.io/
https://github.com/pogrebitskiy/CNN-Adversarial-Attacks
https://github.com/ericyoc/adversarial-defense-cnn-poc/tree/main
https://huggingface.co/timm?search_models=xcit
https://huggingface.co/spaces/timm/leaderboard
https://pytorch.org/vision/stable/transforms.html
    augmentation policies
"""
from utilities import nn_params, plot_accuracy_loss, print_images, save_clean_model
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
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_clean_model(model, save_dir):
    # Model folder
    model_path = f"{save_dir}/clean_model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in {model_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"Loaded clean_model state dict from {model_path}")
    return model

# -----------------------------------------------------------------------------
# Model Zoo
# -----------------------------------------------------------------------------

# Custom CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Input shape: (3, 50, 50)
        self.conv1 = nn.Conv2d(3, 50, 3)
        self.conv2 = nn.Conv2d(50, 40, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(40*46*46, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout1(x)))
        x = self.fc2(self.fc3(self.dropout2(x)))
        
        return F.log_softmax(x, dim=1)


# Cross-Covariance Image Transformer
@register_model
def get_xcit(**kwargs):        
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        eta=1.0,
        tokens_norm=True,
        **kwargs
    )
    model = xcit._create_xcit('xcit_small_12_p4_32', pretrained=False, **model_kwargs)
    assert isinstance(model, xcit.Xcit)
    return model


# ResNet-50 with GELU activation
@register_model
def resnet50_gelu(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with GELU."""
    model_args = dict(block=resnet.Bottleneck,
                      layers=[3, 4, 6, 3],
                      act_layer=lambda inplace: nn.GELU(),
                      **kwargs)
    return resnet._create_resnet('resnet50_gelu', pretrained, **model_args)


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

# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------

# model = get_xcit(img_size=224, num_classes=102)
# model = resnet50_gelu(pretrained=False, num_classes=102)
model = resnet_pytorch(size=18, pretrained=True, num_classes=102)
print(f"{nn_params(model)//1e6:.2f}M")

model.to(device)
col_names = [
    "input_size",
    "output_size",
    # "num_params",
    # "params_percent",
    # "kernel_size",
    # "mult_adds",
    "trainable"
]
print(summary(model, (10, 3, 224, 224), col_names=col_names, verbose=0))
# -----------------------------------------------------------------------------
# Augmentations: Horizontal Flipping, Random Resize-Rescale, Color Jittering
# -----------------------------------------------------------------------------


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.ToTensor(),
    # Flowers102 normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# Dataset and Dataloader: Flowers102
# -----------------------------------------------------------------------------

from torch.utils.data import DataLoader, Dataset, TensorDataset

train_set = torchvision.datasets.Flowers102(root='data', download=True, split='train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

val_set = torchvision.datasets.Flowers102(root='data', download=True, split='val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)

test_set = torchvision.datasets.Flowers102(root='data', download=True, split='test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# -----------------------------------------------------------------------------
# Data visualization
# -----------------------------------------------------------------------------

it = iter(train_loader)
images, labels = next(it)

def plot_tensor_images(images, labels, n):
    fig, ax = plt.subplots(n, n, figsize=(20, 20))
    for i in range(n):
        for j in range(n):
            index = i * n + j
            ax[i, j].imshow(images[index].permute(1, 2, 0))
            ax[i, j].set_title(labels[index].item())
            ax[i, j].axis('off')
    plt.show()

plot_tensor_images(images, labels, 4)

print(model(images.to(device)).shape)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

# TODO: 10-epoch linear epsilon-warmup
optimizer = torch.optim.AdamW(model.parameters())
opt_lr, opt_weight_decay = 1e-4, 0.5
loss_fn = nn.CrossEntropyLoss()     # nn.NLLLoss()


def evaluate(model, data_loader, loss_fn, device):
    test_loss = 0.0
    test_accuracy = 0.0

    test_total = 0
    test_correct = 0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
            x, y = x.to(device), y.to(device)  # Move data to the specified device
            logits = model(x)
            batch_loss = loss_fn(logits, y)
            test_loss += batch_loss.item()
            preds = torch.argmax(logits, dim=1)
            
            test_total += y.size(0)
            test_correct += (preds == y).sum().item()
            
    test_loss /= len(val_loader)
    test_accuracy = test_correct / test_total
    print(f"Test/Val/Attack loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}")

    return test_loss, test_accuracy


def train_model(model, train_loader, loss_fn, optimizer, num_epochs, device, learning_rate=0.001, weight_decay=0.0001):
    # Model folder
    save_dir = 'clean_models'

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    model.to(device)  # Move the model to the specified device

    # Adjust learning rate and weight decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        param_group['weight_decay'] = weight_decay

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (imgs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            imgs, labels = imgs.to(device), labels.to(device)  # Move input data to the specified device

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(preds.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # if i == 0:
            #     print_images(imgs.cpu(), labels.cpu(), preds.cpu(), n=3)  # Move data back to CPU for printing

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Evaluation
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

    # Plot accuracy and loss curves
    plot_accuracy_loss(train_accuracies, val_accuracies, train_losses, val_losses)

    save_clean_model(model, save_dir)


num_epochs = 10
train_model(model, train_loader, loss_fn, optimizer, num_epochs, device, learning_rate=opt_lr, weight_decay=opt_weight_decay)


# -----------------------------------------------------------------------------
# Model evaluation
# -----------------------------------------------------------------------------

model = load_clean_model(model, 'clean_models')

test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)


# -----------------------------------------------------------------------------
# Adversarial attacks
# -----------------------------------------------------------------------------

def compounded_attack(choose_attack_option):
    if choose_attack_option == "fgsm_cw_attack":
        attack1 = torchattacks.FGSM(model, eps=0.3)
        attack2 = torchattacks.CW(model, c=0.1, kappa=0.0, steps=1000)
        attack = torchattacks.MultiAttack([attack1, attack2])
    elif choose_attack_option == "fgsm_pgd_attack":
        attack1 = torchattacks.FGSM(model, eps=0.3)
        attack2 = torchattacks.PGD(model, eps=0.3, alpha=0.01, steps=5)
        attack = torchattacks.MultiAttack([attack1, attack2])
    elif choose_attack_option == "cw_pgd_attack":
        attack1 = torchattacks.CW(model, c=0.1, kappa=0.0, steps=1000)
        attack2 = torchattacks.PGD(model, eps=0.3, alpha=0.01, steps=5)
        attack = torchattacks.MultiAttack([attack1, attack2])
    else:
        print("You did not chose any of the possible options or an you made an invalid option")
    return attack


attack = compounded_attack("fgsm_cw_attack")


def perform_adv_attack(model, attack, data_loader):
    adv_data = []
    adv_labels = []
    device = next(model.parameters()).device

    total_batches = len(data_loader)
    progress_bar = tqdm(total=total_batches, unit='batch', desc='Adversarial Attack')

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        # Convert labels to long data type
        y = y.long()

        x_adv = attack(x, y)
        adv_data.append(x_adv.detach())
        adv_labels.append(y.detach())

        progress_bar.update(1)

    progress_bar.close()

    adv_data = torch.cat(adv_data)
    adv_labels = torch.cat(adv_labels)

    # Move data back to CPU if necessary
    if device != torch.device('cpu'):
        adv_data = adv_data.to('cpu')
        adv_labels = adv_labels.to('cpu')

    # Create dataset and loader
    adv_dataset = TensorDataset(adv_data, adv_labels)
    adv_loader = DataLoader(adv_dataset, batch_size=64, shuffle=False)

    return adv_loader

model = model.to(device)
adv_loader = perform_adv_attack(model, attack, val_loader)

adv_imgs, adv_labels = next(iter(adv_loader))
print_images(adv_imgs, adv_labels, model(adv_imgs.to(device)).detach(), n=3)


train_loss, train_accuracy = evaluate(model, train_loader, loss_fn, device)
val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
adv_loss, adv_accuracy = evaluate(model, adv_loader, loss_fn, device)


# -----------------------------------------------------------------------------
# Adversarial dataset generation
# -----------------------------------------------------------------------------

def generate_adversarial_dataset(model, attack, data_loader, device, num_samples):
    adv_data = []
    adv_labels = []
    device = next(model.parameters()).device

    num_generated = 0
    progress_bar = tqdm(total=num_samples, unit='sample', desc='Adversarial Attack')

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        y = y.long()

        x_adv = attack(x, y)

        with torch.no_grad():
            orig_preds = model(x).argmax(1)
            adv_preds = model(x_adv).argmax(1)

        # Filter samples where original prediction is different from adversarial prediction
        mask = orig_preds != adv_preds
        x_adv_filtered = x_adv[mask]
        y_filtered = y[mask]

        # Append only the required number of samples
        num_remaining = num_samples - num_generated
        if num_remaining > 0:
            if num_remaining >= len(x_adv_filtered):
                adv_data.append(x_adv_filtered.detach())
                adv_labels.append(y_filtered.detach())
                num_generated += len(x_adv_filtered)
                progress_bar.update(len(x_adv_filtered))
            else:
                adv_data.append(x_adv_filtered[:num_remaining].detach())
                adv_labels.append(y_filtered[:num_remaining].detach())
                num_generated = num_samples
                progress_bar.update(num_remaining)
                break

        if num_generated >= num_samples:
            break

    progress_bar.close()

    if len(adv_data) > 0:
        adv_data = torch.cat(adv_data)
        adv_labels = torch.cat(adv_labels)

        if device != torch.device('cpu'):
            adv_data = adv_data.to('cpu')
            adv_labels = adv_labels.to('cpu')

        adv_dataset = TensorDataset(adv_data, adv_labels)

        if num_generated < num_samples:
            print(f"Warning: Could not generate {num_samples} unique adversarial examples.")
            print(f"Returning {num_generated} unique adversarial examples.")
    else:
        adv_dataset = None
        print("Warning: No unique adversarial examples found.")

    return adv_dataset

adv_dataset = generate_adversarial_dataset(model, attack, val_loader, device, num_samples=1000)

# -----------------------------------------------------------------------------
# Combine clean and adversarial datasets
# -----------------------------------------------------------------------------

def combine_datasets(clean_dataset, adv_dataset):
    # Adversarial dataset
    adv_loader = DataLoader(adv_dataset, batch_size=len(adv_dataset))
    adv_images, adv_labels = next(iter(adv_loader))

    # Filtered training set
    images, labels = next(iter(clean_dataset))

    # Combine into one dataset
    combined_tensors = (torch.cat((adv_images, images)),
                        torch.cat((adv_labels, labels)))

    combined_tensor_dataset = TensorDataset(*combined_tensors)
    print(f"Combined dataset size: {len(combined_tensor_dataset)}")

    # Create combined data loader
    combined_loader = DataLoader(combined_tensor_dataset, batch_size=64, shuffle=True)

    return combined_dataset, combined_loader

combined_dataset, combined_loader = combine_datasets(train_loader, adv_dataset)

# -----------------------------------------------------------------------------
# Adversarial training
# -----------------------------------------------------------------------------

model = load_clean_model(model, 'clean_models')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss() 

train_model(model, combined_loader, loss_fn, optimizer, num_epochs, device, learning_rate=opt_lr, weight_decay=opt_weight_decay)


# -----------------------------------------------------------------------------
# Model evaluation
# -----------------------------------------------------------------------------

train_loss, train_accuracy = evaluate(model, train_loader, loss_fn, device)
val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
adv_loss, adv_accuracy = evaluate(model, adv_loader, loss_fn, device)
test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)

