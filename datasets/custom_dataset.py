import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms


class CustomImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    df = pd.read_csv(annotations_file, names=['file_name'])
    df['labels'] = df['file_name'].str.split(' ').str.get(0)
    self.img_labels = df
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
    return image, label


if __name__ == '__main__':
  transform = transforms.Compose([
    transforms.Resize((400, 400,)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  ])


  training_data = CustomImageDataset('lego/validation.txt', 'lego/dataset', transform=transform)
  print(len(training_data))
  print(training_data.img_labels)

  train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
  #test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

  train_features, train_labels = next(iter(train_dataloader))
  print(f'Feature batch shape: {train_features.size()}')
  img = train_features[0].squeeze().permute(1, 2, 0)
  label = train_labels[0]
  plt.imshow(img, cmap='gray')
  plt.show(img)
  print(f'Label: {label}')
