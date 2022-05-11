import torch
import torch.nn as nn
from torch import optim
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
from torchsummary import summary

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def get_overview():
  # load FMNIST dataset
  data_folder = '../../data'
  fmnist = datasets.FashionMNIST(data_folder,
                                 download=True,
                                 train=True)
  tr_images = fmnist.data
  tr_targets = fmnist.targets

  # overview of dataset
  unique_values = tr_targets.unique()
  print(f'tr_images & tr_targets:\n\tX - {tr_images.shape}\n\tY\
   - {tr_targets.shape}\n\tY - Unique Values : {unique_values}')
  print(f'TASK:\n\t{len(unique_values)} class Classification')
  print(f'UNIQUE CLASSES: \n\t{fmnist.classes})')

  # visual overview
  R, C = len(tr_targets.unique()), 10
  fig, ax = plt.subplots(R, C, figsize=(10,10))
  for label_class, plot_row in enumerate(ax):
    label_x_rows = np.where(tr_targets == label_class)[0]

    for plot_cell in plot_row:
      plot_cell.grid(False)
      plot_cell.axis('off')
      idx = np.random.choice(label_x_rows)
      x, y = tr_images[idx], tr_targets[idx]
      plot_cell.imshow(x, cmap='gray')
  plt.tight_layout()
  plt.show()


def visualize(train_losses, train_accuracies, val_losses, val_accuracies, epochs, batch_size, lr):
  epochs = np.arange(epochs)+1

  plt.subplot(211)
  plt.plot(epochs, train_losses, 'bo', label='Training loss')
  plt.plot(epochs, val_losses, 'r', label='Validation loss')
  plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
  plt.title(f'Training and validation loss when batch size: {batch_size}, lr: {lr}')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid('off')

  plt.subplot(212)
  plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
  plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
  plt.title(f'Training and validation accuracy when batch size: {batch_size}, lr: {lr}')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
  plt.legend()
  plt.grid('off')
  plt.show()


def print_distributions(model):
  for idx, param in enumerate(model.parameters()):
    if idx == 0:
      plt.hist(param.cpu().detach().numpy().flatten())
      plt.title('Distribution of weights connecting input to hidden layer')
      plt.show()
    elif idx == 1:
      plt.hist(param.cpu().detach().numpy().flatten())
      plt.title('Distribution of biases of hidden layer')
      plt.show()
    elif idx == 2:
      plt.hist(param.cpu().detach().numpy().flatten())
      plt.title('Distribution of weights connecting hidden to output layer')
      plt.show()
    elif idx == 3:
      plt.hist(param.cpu().detach().numpy().flatten())
      plt.title('Distribution of biases of output layer')
      plt.show()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_folder = '../../data'

# train dataset
fmnist = datasets.FashionMNIST(data_folder,
                               download=True,
                               train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

# validation dataset
val_fmnist = datasets.FashionMNIST(data_folder,
                                   download=True,
                                   train=False)
val_images = val_fmnist.data
val_targets = val_fmnist.targets


class MyNeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.dropout1 = nn.Dropout(0.25)
    self.input_to_hidden_layer = nn.Linear(28*28, 1000)
    self.batch_norm = nn.BatchNorm1d(1000)
    self.hidden_layer_activation = nn.ReLU()
    self.dropout2 = nn.Dropout(0.25)
    self.hidden_to_output_layer = nn.Linear(1000, 10)

  def forward(self, x):
    x = self.input_to_hidden_layer(x)
    x0 = self.batch_norm(x)
    x1 = self.hidden_layer_activation(x0)
    x2 = self.hidden_to_output_layer(x1)
    return x2, x1


class FMNISTDataset(Dataset):
  def __init__(self, x, y):
    x = x.float()/255
    x = x.view(-1, 28*28)
    self.x = x
    self.y = y

  def __getitem__(self, idx):
    x, y = self.x[idx], self.y[idx]
    return x.to(device), y.to(device)

  def __len__(self):
    return len(self.x)


def get_data(batch_size):
  train = FMNISTDataset(tr_images, tr_targets)
  trn_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
  val = FMNISTDataset(val_images, val_targets)
  val_dl = DataLoader(val, batch_size=len(val_images), shuffle=False)
  return trn_dl, val_dl


def get_model(learning_rate):
  #model = nn.Sequential(nn.Linear(28*28, 1000),
  #                      nn.ReLU(),
  #                      nn.Linear(1000, 10)).to(device)
  model = MyNeuralNet().to(device)
  loss_func = nn.CrossEntropyLoss()
  optimizer = SGD(model.parameters(), lr=learning_rate)
  return model, loss_func, optimizer


def train_batch(x, y, model, opt, loss_fn):
  model.train()
  prediction = model(x)[0]

  #l1_regularization = 0
  l2_regularization = 0
  for param in model.parameters():
    #l1_regularization += torch.norm(param, 1)
    l2_regularization += torch.norm(param, 2)
  #batch_loss = loss_fn(prediction, y)+0.0001*l1_regularization
  batch_loss = loss_fn(prediction, y)+0.01*l2_regularization

  batch_loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
  model.eval()
  with torch.no_grad():
    prediction = model(x)[0]
  max_values, argmaxes = prediction.max(-1)
  is_correct = argmaxes == y
  return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(x, y, model):
  model.eval()
  prediction = model(x)[0]
  val_loss = loss_fn(prediction, y)
  return val_loss.item()
  

epochs = 30
batch_size = 32
learning_rate = 1e-3
trn_dl, val_dl = get_data(batch_size)
model, loss_fn, optimizer = get_model(learning_rate)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.5,
                                                 patience=0,
                                                 threshold=0.001,
                                                 verbose=True,
                                                 min_lr=1e-5,
                                                 threshold_mode='abs')

for epoch in range(epochs):
  print(f'--- Epoch: {epoch} ---')

  # training
  train_epoch_losses, train_epoch_accuracies = [], []
  for idx, batch in enumerate(iter(trn_dl)):
    x, y = batch
    batch_loss = train_batch(x, y, model, optimizer, loss_fn)
    train_epoch_losses.append(batch_loss)
  train_epoch_loss = np.array(train_epoch_losses).mean()

  # accuracy
  for idx, batch in enumerate(iter(trn_dl)):
    x, y = batch
    is_correct = accuracy(x, y, model)
    train_epoch_accuracies.extend(is_correct)
  train_epoch_accuracy = np.mean(train_epoch_accuracies)

  # validation
  for idx, batch in enumerate(iter(val_dl)):
    x, y = batch
    val_is_correct = accuracy(x, y, model)
    validation_loss = val_loss(x, y, model)
    scheduler.step(validation_loss)
  val_epoch_accuracy = np.mean(val_is_correct)

  # store losses and accuracies
  train_losses.append(train_epoch_loss)
  train_accuracies.append(train_epoch_accuracy)
  val_losses.append(validation_loss)
  val_accuracies.append(val_epoch_accuracy)

torch.save(model.to('cpu').state_dict(), 'fmnist.pth')
#print_distributions(model)

visualize(train_losses, train_accuracies, val_losses, val_accuracies, epochs, batch_size, learning_rate)
