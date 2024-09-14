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

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def nn_params(model):
    """Get the number of parameters of a model

    References:
        https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_accuracy_loss(train_accuracies, val_accuracies, train_losses, val_losses):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_images(images, labels, preds, n):

    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            index = i * n + j
            # Normalize the image data
            image = images[index].permute(1, 2, 0)
            image = (image - image.min()) / (image.max() - image.min())
            ax[i, j].imshow(image)
            pred = preds[i].argmax().item()
            ax[i, j].set_title(f"{labels[index].item()} -> {pred}")
            ax[i, j].axis('off')
    plt.show()


def save_clean_model(model, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = f"{save_dir}/clean_model.pth"

    torch.save(model.state_dict(), model_path)


# -----------------------------------------------------------------------------
# Analyze dataset
# -----------------------------------------------------------------------------

train_dataset = pd.concat([
    pd.read_csv('./data/train_1.csv').assign(dataset='train_1'),
    pd.read_csv('data/train_2.csv').assign(dataset='train_2'),
    pd.read_csv('data/train_3.csv').assign(dataset='train_3'),
], ignore_index=True)

print(train_dataset.shape)

train_dataset['ImageId'] = train_dataset['ImageId'].apply(lambda x: x + '.png' if not x.endswith('.png') else x)
train_dataset['Path'] = train_dataset.apply(
    lambda x: f'data/NIVEL1/NIVEL1/TRAIN/{x["dataset"]}/{x["ImageId"]}', axis=1
)

train_1 = train_dataset[train_dataset['dataset'] == 'train_1']
train_2 = train_dataset[train_dataset['dataset'] == 'train_2']
train_3 = train_dataset[train_dataset['dataset'] == 'train_3']

def split_data(data):
    train, val = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['Label'])
    return train, val
train_1_train, train_1_val = split_data(train_1)
train_2_train, train_2_val = split_data(train_2)
train_3_train, train_3_val = split_data(train_3)

train_df = pd.concat([train_1_train, train_2_train, train_3_train]).reset_index(drop=True)
val_df = pd.concat([train_1_val, train_2_val, train_3_val]).reset_index(drop=True)

print(len(train_dataset))
print(len(train_df), len(val_df))

# -----------------------------------------------------------------------------
# Plot example of each dataset
# -----------------------------------------------------------------------------

# C:\Users\jeff1\Desktop\dl-playground\dl-utec\data\NIVEL1\NIVEL1\TRAIN\train_1\c_12993.png
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, dataset in enumerate(train_dataset['dataset'].unique()):
    img_row = train_dataset[train_dataset['dataset'] == dataset].sample(1)
    # print(img_row['Path'].values[0])
    # img_path = f"data/NIVEL1/NIVEL1/TRAIN/{img_row[2]}/{img_row[0]}"
    img_path = img_row['Path'].values[0]
    # print(img_path)
    image = plt.imread(img_path)
    axs[i].imshow(image)
    axs[i].set_title(dataset + ' - ' + image.shape.__str__())
    axs[i].axis('off')
plt.show()


# -----------------------------------------------------------------------------
# Create dataset
# -----------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    # transforms.RandomResizedCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    # TODO: CIFAR10, MNIST, FASHION MNIST Normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    # transforms.RandomResizedCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    # TODO: CIFAR10, MNIST, FASHION MNIST Normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Nivel1Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_row = self.dataframe.iloc[idx]
        # img_path = f"data/NIVEL1/NIVEL1/TRAIN/{img_row[2]}/{img_row[0]}"
        img_path = img_row['Path']

        image = plt.imread(img_path)
        label = self.dataframe.iloc[idx, 1]

        # * if image only have 1 channel, convert to 3 channels
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        # print(img_path, type(image), image.shape, label)

        # * apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, label

train_set = Nivel1Dataset(train_df.sample(frac=1).reset_index(drop=True), transform=transform)
val_set = Nivel1Dataset(val_df, transform=transform_val)
# test_set = Nivel1Dataset(test_dataset, transform=transform)
print(train_set[0])
print(val_set[0])

# print(test_set[0])


# -----------------------------------------------------------------------------
# Create data loaders
# -----------------------------------------------------------------------------

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)

print(len(train_loader), len(val_loader))

img, label = next(iter(train_loader))
print(img.shape, label.shape)

for data in train_loader:
    imgs, labels = data
    print(imgs.shape, labels.shape)
    break

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

    # # Freeze all layers except the output layer
    # for name, param in model.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False
    
    return model


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 30)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



model = resnet_pytorch(size=18, pretrained=True, num_classes=30)
# model = Net()
print(f"{nn_params(model)//1e6:.2f}M")


# -----------------------------------------------------------------------------
# Train model
# -----------------------------------------------------------------------------

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


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, learning_rate=0.001, weight_decay=0.0001):
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


num_epochs = 1
train_model(model, train_loader, val_loader, 
            loss_fn, optimizer, num_epochs, device, 
            learning_rate=opt_lr, weight_decay=opt_weight_decay)


# -----------------------------------------------------------------------------
# Inference and Submission Generation
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
        image = transform_val(image).unsqueeze(0).to(device)
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


tags = 'last'
torch.save(model.state_dict(), f'model_{tags}.pth')
generate_submission(model, 'test1', './data/NIVEL1/NIVEL1/TEST1_3/**/*.png', tags=tags)
generate_submission(model, 'test2', './data/NIVEL2/NIVEL2/TEST2_3/**/*.png', tags=tags)
generate_submission(model, 'test3', './data/NIVEL3/NIVEL3/TEST3_3/**/*.png', tags=tags)



