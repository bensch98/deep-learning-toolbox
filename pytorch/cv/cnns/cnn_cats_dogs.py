import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torchvision import transforms, models, datasets
from torchsummary import summary

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from glob import glob
from random import shuffle, seed

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

seed(420)

# not enough GPU memory -> force to use CPU instead of cuda
torch.cuda.is_available = lambda: False
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize(epochs, train_accuracies, val_accuracies):
  epochs = np.arange(epochs)+1

  plt.plot(epochs, train_accuracies, 'bo', label='Trainig accuracy')
  plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
  plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
  plt.title('Training and validation accuracy with 4K data points used for training')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.gca().set_yticklabels(['{:.0f}'.format(x*100) for x in plt.gca().get_yticks()])
  plt.legend()
  plt.grid('off')
  plt.show()


class CatDogDataset(Dataset):
  def __init__(self, folder):
    cats = glob(folder+'/cats/*.jpg')
    dogs = glob(folder+'/dogs/*.jpg')
    self.fpaths = cats + dogs
    shuffle(self.fpaths)
    self.targets = [fpath.split('/')[-1].startswith('dog') for fpath in self.fpaths]

  def __len__(self):
    return len(self.fpaths)

  def __getitem__(self, idx):
    f = self.fpaths[idx]
    target = self.targets[idx]
    img = (cv2.imread(f)[:,:,::-1])
    img  = cv2.resize(img, (224,224))
    return torch.tensor(img/255).permute(2,0,1).to(device).float(), \
           torch.tensor([target]).float().to(device)


class DogCatConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = conv_layer(3, 64, 3)
    self.conv2 = conv_layer(64, 512, 3)
    self.conv3 = conv_layer(512, 512, 3)
    self.conv4 = conv_layer(512, 512, 3)
    self.conv5 = conv_layer(512, 512, 3)
    self.conv6 = conv_layer(512, 512, 3)
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(512, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.flatten(x)
    x = self.linear(x)
    x = self.sigmoid(x)
    return x


def conv_layer(ni, no, kernel_size, stride=1):
  return nn.Sequential(
    nn.Conv2d(ni, no, kernel_size, stride),
    nn.ReLU(),
    nn.BatchNorm2d(no),
    nn.MaxPool2d(2),
  )


def get_model():
  model = DogCatConvNet().to(device)
  loss_fn = nn.BCELoss()
  opt = Adam(model.parameters(), lr=1e-3)
  return model, loss_fn, opt


def get_data():
  train_data_dir = '../datasets/cat-and-dog/training_set/training_set'
  test_data_dir = '../datasets/cat-and-dog/test_set/test_set'

  # inspect random image
  train = CatDogDataset(train_data_dir)
  trn_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
  val = CatDogDataset(test_data_dir)
  val_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
  return trn_dl, val_dl


def train_batch(x, y, model, opt, loss_fn):
  model.train()
  prediction = model(x)
  batch_loss = loss_fn(prediction, y)
  batch_loss.backward()
  opt.step()
  opt.zero_grad()
  return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
  prediction = model(x)
  is_correct = (prediction > 0.5) == y
  return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(x, y, model, loss_fn):
  prediction = model(x)
  val_loss = loss_fn(prediction, y)
  return val_loss.item()


if __name__ == '__main__':
  # hyperparameters
  epochs = 5

  # setup model
  trn_dl, val_dl = get_data()
  model, loss_fn, opt = get_model()
  summary(model, torch.zeros(1,3,224,224))

  train_losses, train_accuracies = [], []
  val_losses, val_accuracies = [], []

  for epoch in range(epochs):
    print(f'--- Epoch: {epoch} ---')
    train_epoch_losses, train_epoch_accuracies = [], []
    val_epoch_losses = []

    for idx, batch in enumerate(iter(trn_dl)):
      x, y = batch
      batch_loss = train_batch(x, y, model, opt, loss_fn)
      train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()

    for idx, batch in enumerate(iter(trn_dl)):
      x, y = batch
      is_correct = accuracy(x, y, model)
      train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)

    for idx, batch in enumerate(iter(val_dl)):
      x, y = batch
      val_is_correct = accuracy(x, y, model)
      val_epoch_accuracies.extend(val_is_correct)
    val_epoch_accuracy = np.mean(val_epoch_accuracies)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_accuracies.append(val_epoch_accuracy)

  torch.save(model.to('cpu').state_dict(), '../models/cnn_catdog.pth')
  visualize(epochs, train_accuracies, val_accuracies)
