import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam

from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from imgaug import augmenters as iaa
from copy import deepcopy


# not enough gpu memory -> force to use cpu instead of cuda
torch.cuda.is_available = lambda: False
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize(epochs, train_losses, val_losses, train_accuracies, val_accuracies):
  epochs = np.arange(epochs)+1

  # subplot 1
  plt.subplot(211)
  plt.plot(epochs, train_losses, 'bo', label='Training loss')
  plt.plot(epochs, val_losses, 'r', label='Validation loss')
  plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
  plt.title('Training and validation loss with CNN')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid('off')

  # subplot 2
  plt.subplot(212)
  plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
  plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
  plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
  plt.title('Training and validation accuracy with CNN')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
  plt.legend()
  plt.grid('off')

  plt.show()


class FMNISTDataset(Dataset):
  def __init__(self, data, transform=None):
    x = data.data
    y = data.targets
    x = x.float()/255
    x = x.view(-1,1,28,28)
    self.x = x
    self.y = y
    self.transform = transform

  def __getitem__(self, idx):
    x, y = self.x[idx], self.y[idx]
    if self.transform:
      x = self.transform(x)
      y = self.transform(y)
    return x, y

  def __len__(self):
    return len(self.x)


class MyConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
    self.maxpool1 = nn.MaxPool2d(2)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
    self.maxpool2 = nn.MaxPool2d(2)
    self.relu2 = nn.ReLU()
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(800, 64)
    self.relu3 = nn.ReLU()
    self.linear2 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.maxpool2(x)
    x = self.relu2(x)
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.relu3(x)
    x = self.linear2(x)
    return x


def get_model():
  model = MyConvNet().to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=1e-3)
  return model, loss_fn, optimizer


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
  model.eval()
  prediction = model(x)
  max_values, argmaxes = prediction.max(-1)
  is_correct = argmaxes == y
  return is_correct.cpu().numpy().tolist()


def get_data(train_data, val_data, aug):
  train = FMNISTDataset(train_data, transform=aug)
  trn_dl = DataLoader(train, batch_size=2, shuffle=True)
  val = FMNISTDataset(val_data)
  val_dl = DataLoader(val, batch_size=len(val_data), shuffle=True)
  return trn_dl, val_dl


@torch.no_grad()
def val_loss(x, y, model, loss_fn):
  model.eval()
  prediction = model(x)
  val_loss = loss_fn(prediction, y)
  return val_loss.item()


if __name__ == '__main__':
  torch.cuda.empty_cache()
  data_folder = '../../../data'
  
  # fetch data
  fmnist = datasets.FashionMNIST(data_folder,
                                 download=True,
                                 train=True)
  val_fmnist = datasets.FashionMNIST(data_folder,
                                     download=True,
                                     train=False)
  # split dataset
  #tr_images = fmnist.data
  #tr_targets = fmnist.targets
  #val_images = val_fmnist.data
  #val_targets = val_fmnist.targets

  # augmentation pipeline
  aug = transforms.Compose([
  ])
  #aug = torch.jit.script(aug)

  # init data loaders
  trn_dl, val_dl = get_data(fmnist, val_fmnist, aug)

  # setup model training
  model, loss_fn, opt = get_model()
  train_losses, train_accuracies = [], []
  val_losses, val_accuracies = [], []

  summary(model, torch.zeros(1,1,28,28))

  epochs = 5

  # *** training ***
  for epoch in range(epochs):
    print(f'--- Epoch: {epoch} ---')
    #print(torch.cuda.memory_summary('cuda', True))
    train_epoch_losses, train_epoch_accuracies = [], []
    
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
      validation_loss = val_loss(x, y, model, loss_fn)
    val_epoch_accuracy = np.mean(val_is_correct)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(validation_loss)
    val_accuracies.append(val_epoch_accuracy)

  torch.save(model.to('cpu').state_dict(), '../models/cnn_fmnist.pth')
  visualize(epochs, train_losses, train_accuracies, val_losses, val_accuracies)
