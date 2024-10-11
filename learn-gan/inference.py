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


import timm
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

N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0


# %%
# -----------------------------------------------------------------------------
# Import model ckpt
# -----------------------------------------------------------------------------

from utils import gradient_penalty, save_checkpoint, load_checkpoint
from models.wgan_gp import Discriminator, Generator, initialize_weights

CKPT_PATH = "sub_ckpt.pth.tar"
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)


if (device.type == 'cuda') and (N_GPU > 1):
    gen = nn.DataParallel(gen, list(range(N_GPU)))


initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

load_checkpoint(torch.load(CKPT_PATH), gen, critic)

# %%
# -----------------------------------------------------------------------------
# Smooth image interpolation/sampling
# -----------------------------------------------------------------------------


# %%
# -----------------------------------------------------------------------------
# Submission generation
# -----------------------------------------------------------------------------

