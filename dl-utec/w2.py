# %matplotlib inline

import torch
from torchvision import utils
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from tqdm import tqdm

torch.random.manual_seed(0)

DATASET = 'CIFAR10'
BATCH_SIZE = 4
LOSS_FUNCTION = 'CELoss'
SAVE_PATH = './results/'
os.makedirs(SAVE_PATH, exist_ok=True)

# -----------------------------------------------------------------------------
# Transformation, Dataset and DataLoader
# -----------------------------------------------------------------------------

if DATASET == 'MNIST':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
elif DATASET == 'CIFAR10':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True) # , num_workers=2
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False) # , num_workers=2

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_accuracy_global_and_per_class(net, testloader, verbose=False):
    response = {}
    # * GLOBAL ACCURACY
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    global_accuracy = 100 * correct / total
    response["global_accuracy"] = global_accuracy
    if verbose:
        print(f'Accuracy of the network on the 10000 test images: {global_accuracy} %')

    # * PER CLASS ACCURACY
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    response["per_class_accuracy"] = {}
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        if verbose:
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        response["per_class_accuracy"][classname] = accuracy

    return response


def vis_tensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    # https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n*c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


# -----------------------------------------------------------------------------
# Show sample dataset
# -----------------------------------------------------------------------------

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        if DATASET == 'MNIST':
            self.conv1 = nn.Conv2d(1, 6, 5)
        elif DATASET == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 16, 5)
        if DATASET == 'MNIST':
            in_features_ = 256
        elif DATASET == 'CIFAR10':
            in_features_ = 16 * 5 * 5
        self.fc1 = nn.Linear(in_features_, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        return x

net = Net()
net = torchvision.models.resnet18(pretrained=True)
print(net)
net(images)

# -----------------------------------------------------------------------------
# Loss Function and Optimizer
# -----------------------------------------------------------------------------


if LOSS_FUNCTION == 'CELoss':
    criterion = nn.CrossEntropyLoss()
elif LOSS_FUNCTION == 'MSELoss':
    criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=0.0001, gamma=0.1)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

train_loss = []
test_accuracy = []
epoch_id = []
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        # print statistics
        running_loss += loss.item()

        if i % 2000 == 1999:    # print every 2000 mini-batches
            avg_loss = running_loss / 2000
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
            train_loss.append(avg_loss)
            test_accuracy.append(get_accuracy_global_and_per_class(net, testloader)["global_accuracy"])
            epoch_id.append(epoch + i / len(trainloader))
            running_loss = 0.0

print('Finished Training')

# -----------------------------------------------------------------------------
# Save model
# -----------------------------------------------------------------------------

PATH = f'./{SAVE_PATH}/net_{DATASET}_{LOSS_FUNCTION}.pth'
torch.save(net.state_dict(), PATH)

# -----------------------------------------------------------------------------
# Test the network on the test data
# -----------------------------------------------------------------------------

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------

net = torchvision.models.resnet18(pretrained=False)
net.load_state_dict(torch.load(PATH))


# -----------------------------------------------------------------------------
# Test model
# -----------------------------------------------------------------------------

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))


# -----------------------------------------------------------------------------
# Get model general and per class accuracy
# -----------------------------------------------------------------------------

accuracy_dict = get_accuracy_global_and_per_class(net, testloader, True)
accuracy_dict["train_loss"] = train_loss
accuracy_dict["test_accuracy"] = test_accuracy
accuracy_dict["epoch_id"] = epoch_id
print(json.dumps(accuracy_dict, indent=4))


# -----------------------------------------------------------------------------
# Visualize kernels
# -----------------------------------------------------------------------------


filter = net.conv1.weight.data.clone()
vis_tensor(filter, ch=3, allkernels=False)

plt.axis('off')
plt.ioff()
plt.show()

