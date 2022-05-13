import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


class ComponentDataset(Dataset):
  def __init__(self, data, path, transform=None):
    self.data = data.name.tolist()
    self.label = data.label.tolist()
    self.path = path
    self.transform = transform

  def __getitem__(self, idx):
    img_name, label = self.data[idx], self.label[idx]
    img_path = f'{self.path}/Figure_{str(idx+1)}.png'
    img = Image.open(img_path)
    img = img.convert('RGB')
    if self.transform:
      img = self.transform(img)
    return img, label

  def __len__(self):
    return len(self.data)


if __name__ == '__main__':
  data_folder = '../data'

  '''
  aug = transforms.Compose([
    transforms.CenterCrop((10, 10)),
    transforms.GaussianBlur(3, (0.1, 2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
  ])

  df = pd.DataFrame({'name': ['Component_1.png'], 'label': [1]})
  #img = f'{data_folder}/Psychedelic_Symphony.jpeg'
  #df = pd.DataFrame({'name': [img], 'label': [1]})
  ds = ComponentDataset(df, data_folder, transform=aug)

  dl = DataLoader(ds, batch_size=1, shuffle=True)
  
  for idx, batch in enumerate(iter(dl)):
    x, y = batch
    print(x.size())
    print(x)

  '''
  #img = f'{data_folder}/Figure_1.png'
  img = f'{data_folder}/Psychedelic_Symphony.jpeg'
  img = Image.open(img)
  print(img.size)
  
  aug = transforms.Compose([
    transforms.CenterCrop((10, 10)),
    transforms.GaussianBlur(3, (0.1, 2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
  ])
  img = aug(img)
  print(img)
  aug = transforms.Compose([
    transforms.ToPILImage(),
  ])
  img = aug(img)
  plt.imshow(img)
  plt.show()
