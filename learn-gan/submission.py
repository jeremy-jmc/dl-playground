from super_resolution.model import Generator, Discriminator
from super_resolution.data.utils import DataSet
from torch.utils.data import DataLoader
import torch 
import os, glob
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, ToPILImage
from tqdm import tqdm
import pandas as pd
import math
from IPython.display import display

generator = Generator(in_channels=3, hid_channels=64, out_channels=3)
discriminator=Discriminator(in_channels=3, hid_channels=64, out_channels=1)

generator.load_state_dict(torch.load('./generator_noise_12.pt'))
discriminator.load_state_dict(torch.load('./discriminator_noise_12.pt'))


test_images = glob.glob('./data/TEST/TEST/*.png')

HD_SIZE = [96, 96]
transformHD = Compose([
    Resize(HD_SIZE),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = DataSet(path='./data/TEST/TEST/', input_size=(24, 24), transform=transformHD)
print(dataset[0].shape)

dataloader = DataLoader(dataset, batch_size=256, shuffle=False)


tensor_results = []
discriminator.eval()
for i, batch in tqdm(iterable=enumerate(dataloader), total=len(dataloader)):
    # print(batch.shape)
    tensor_results.extend(list(discriminator(batch).cpu().detach().numpy().flatten()))

print(tensor_results)

df = pd.DataFrame(
    {
        'Filename': test_images,
        'Score': tensor_results
    }   
)
df['Filename'] = df['Filename'].apply(lambda x: os.path.basename(x))
df['Score'] = df['Score']
df['Prob'] = df['Score'].apply(lambda x: 1 / (1 + math.exp(-x)))

threshold = 0.5
df['Real'] = df['Prob'].apply(lambda x: 1 if x > threshold else 0)

display(df)

print(df['Score'].describe())
print(df['Real'].value_counts(normalize=True))
df[['Filename', 'Real']].to_csv('submission.csv', index=False)

