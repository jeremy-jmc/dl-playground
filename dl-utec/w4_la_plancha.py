# !python -m pip install timm torchinfo
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
"""
from utilities import nn_params, plot_accuracy_loss, print_images, save_clean_model
import torch
import torch.nn as nn
from torchinfo import summary
import torchvision
import timm
from timm.models import xcit, resnet, convnext
from timm.models.registry import register_model
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Model Zoo
# -----------------------------------------------------------------------------

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

@register_model
def resnet50_gelu(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with GELU."""
    model_args = dict(block=resnet.Bottleneck,
                      layers=[3, 4, 6, 3],
                      act_layer=lambda inplace: nn.GELU(),
                      **kwargs)
    return resnet._create_resnet('resnet50_gelu', pretrained, **model_args)


# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------

model = get_xcit(img_size=224, num_classes=102)
# model = resnet50_gelu(pretrained=False, num_classes=102)
print(f"{nn_params(model)//1e6:.2f}M")

model.to(device)
print(summary(model, (10, 3, 224, 224), verbose=0))

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

train_set = torchvision.datasets.Flowers102(root='data', download=True, split='train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

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


def train_model(model, train_loader, loss_fn, optimizer, num_epochs, device, learning_rate=0.001, weight_decay=0.0001):
    # Model folder
    save_dir = 'clean_models'

    train_accuracies = []
    train_losses = []

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

            if i == 0:
                print_images(imgs.cpu(), labels.cpu(), preds.cpu(), n=3)  # Move data back to CPU for printing

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

    # Plot accuracy and loss curves
    plot_accuracy_loss(train_accuracies, train_losses)

    save_clean_model(model, save_dir)


num_epochs = 1
train_model(model, train_loader, loss_fn, optimizer, num_epochs, device, learning_rate=opt_lr, weight_decay=opt_weight_decay)