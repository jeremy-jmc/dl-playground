# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchinfo import summary


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
import warnings
import random
from IPython.display import display
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------

LEARNING_RATE = 1e-4
BATCH_SIZE = 256
IMAGE_SIZE = 32
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# DATA_PATH = '/kaggle/input/2-deep-learning-cs-5364-lab-1-nivel-1'
DATA_PATH = './data'
CKPT_PATH = "./ckpts/checkpoint_epoch.pth.tar"

N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0

# %%
# -----------------------------------------------------------------------------
# Analyze dataset
# -----------------------------------------------------------------------------

train_dataset = pd.concat([
    pd.read_csv(f'{DATA_PATH}/train_1.csv').assign(dataset='train_1'),
    # pd.read_csv(f'{DATA_PATH}/train_2.csv').assign(dataset='train_2'),
    # pd.read_csv(f'{DATA_PATH}/train_3.csv').assign(dataset='train_3'),
], ignore_index=True)

print(train_dataset.shape)

train_dataset['ImageId'] = train_dataset['ImageId'].apply(lambda x: x + '.png' if not x.endswith('.png') else x)
train_dataset['Path'] = train_dataset.apply(
    lambda x: f'{DATA_PATH}/NIVEL1/NIVEL1/TRAIN/{x["dataset"]}/{x["ImageId"]}', axis=1
)

train_1 = train_dataset[train_dataset['dataset'] == 'train_1']
train_2 = train_dataset[train_dataset['dataset'] == 'train_2']
train_3 = train_dataset[train_dataset['dataset'] == 'train_3']

print(train_dataset['Label'].value_counts().sort_index())
# print(train_dataset['Label'].value_counts(normalize=True).sort_index())

def split_data(data):
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    train, val = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['Label'])
    return train, val

# each folder is one of MNIST, CIFAR10, FASHION MNIST
train_1_train, train_1_val = split_data(train_1)
train_2_train, train_2_val = split_data(train_2)
train_3_train, train_3_val = split_data(train_3)

train_df = pd.concat([train_1_train, train_2_train, train_3_train]).reset_index(drop=True)
val_df = pd.concat([train_1_val, train_2_val, train_3_val]).reset_index(drop=True)

print(len(train_dataset))
print(len(train_df), len(val_df))


# -----------------------------------------------------------------------------
# Plot sample of each dataset
# -----------------------------------------------------------------------------

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, dataset in enumerate(train_dataset['dataset'].unique()):
    img_row = train_dataset[train_dataset['dataset'] == dataset].sample(1)
    # print(img_row['Path'].values[0])
    # img_path = f"{DATA_PATH}/NIVEL1/NIVEL1/TRAIN/{img_row[2]}/{img_row[0]}"
    img_path = img_row['Path'].values[0]
    img_label = img_row['Label'].values[0]
    # print(img_path)
    image = plt.imread(img_path)
    axs[i].imshow(image)
    axs[i].set_title(dataset + ' - ' + image.shape.__str__())
    axs[i].axis('off')
plt.show()


# %%
# -----------------------------------------------------------------------------
# Create dataset
# -----------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # transforms.RandomResizedCrop(IMAGE_SIZE),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    # TODO: CIFAR10, MNIST, FASHION MNIST Normalization
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # transforms.RandomResizedCrop(IMAGE_SIZE),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    # TODO: CIFAR10, MNIST, FASHION MNIST Normalization
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class NivelDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_row = self.dataframe.iloc[idx]
        # img_path = f"{DATA_PATH}/NIVEL1/NIVEL1/TRAIN/{img_row[2]}/{img_row[0]}"
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

train_set = NivelDataset(train_df.sample(frac=1).reset_index(drop=True), transform=transform)
# test_set = Nivel1Dataset(test_dataset, transform=transform)
val_set = NivelDataset(val_df, transform=transform_val)

print(train_set[0])
# print(test_set[0])
print(val_set[0])

# %% 

# -----------------------------------------------------------------------------
# Create data loaders
# -----------------------------------------------------------------------------

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print(len(train_loader), len(val_loader))

img, label = next(iter(train_loader))
print(img.shape, label.shape)

for data in train_loader:
    imgs, labels = data
    print(imgs.shape, labels.shape)
    break



# %% 

# -----------------------------------------------------------------------------
# Create model
# -----------------------------------------------------------------------------

from utils import gradient_penalty, save_checkpoint
from models.wgan_gp import Discriminator, Generator, initialize_weights

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)


if (device.type == 'cuda') and (N_GPU > 1):
    gen = nn.DataParallel(gen, list(range(N_GPU)))


initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

gen.train()
critic.train()

# %% 

# -----------------------------------------------------------------------------
# Train model
# -----------------------------------------------------------------------------

loader = train_loader

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)

            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            loss_normal = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        loss_tot = loss_critic + loss_gen
        # Print losses occasionally and print to tensorboard
        if batch_idx % 25 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  loss D: {loss_normal:.4f}, loss G: {loss_gen:.4f}, loss T: {loss_tot:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                # write losses to tensorboard
                writer_real.add_scalar("loss_critic", loss_normal, global_step=step)
                writer_fake.add_scalar("loss_gen", loss_gen, global_step=step)
                writer_fake.add_scalar("loss_tot", loss_tot, global_step=step)
            step += 1

    save_checkpoint({
        'gen': gen.state_dict(),
        'disc': critic.state_dict(),
        'gen_opt': opt_gen.state_dict(),
        'disc_opt': opt_critic.state_dict()
    }, CKPT_PATH.replace('epoch', f'{epoch}_epoch'))


# load_checkpoint(torch.load(CKPT_PATH), gen, critic)

