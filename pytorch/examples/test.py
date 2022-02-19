import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F

from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.conv2 = nn.Conv2d(6, 12, 5)
    self.conv3 = nn.Conv2d(12, 24, 5)
    self.fc1 = nn.Linear(24*2*2, 120)
    self.drop1 = nn.Dropout2d(p=0.2)
    self.fc2 = nn.Linear(120, 60)
    self.fc3 = nn.Linear(60, 10)
    '''
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10),
    )
    '''

  def forward(self, xb):
    # (2) hidden conv layer
    xb = self.conv1(xb)
    xb = F.relu(xb)
    xb = F.max_pool2d(xb, kernel_size=2, stride=2)

    # (3) hidden conv layer
    xb = self.conv2(xb)
    xb = F.relu(xb)
    xb = self.conv3(xb)
    xb = F.relu(xb)
    xb = F.max_pool2d(xb, kernel_size=2, stride=2)

    # (4) hidden linear layer
    xb = xb.reshape(-1, 24 * 2 * 2)
    xb = self.fc1(xb)
    xb = F.relu(xb)

    # dropout layer
    #xb = F.dropout(xb, p=0.5, training=True, inplace=False)
    m = self.drop1(xb)

    # (5) hidden linear layer
    xb = self.fc2(xb)
    xb = F.relu(xb)

    # (6) output layer
    xb = self.fc3(xb)
    return xb

    #return xb.view(-1, xb.size(1))
    '''
    xb = self.flatten(xb)
    logits = self.linear_relu_stack(xb)
    return logits
    '''


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
  for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
      pred = model(xb)
      loss = loss_func(pred, yb)

      loss.backward()
      opt.step()
      opt.zero_grad()
      
    model.eval()
    with torch.no_grad():
      valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))


def preprocess(x, y):
  return x.view(-1, 1, 28, 28).to(device), y.to(device)

class WrappedDataLoader:
  def __init__(self, dl, func):
    self.dl = dl
    self.func = func

  def __len__(self):
    return len(self.dl)

  def __iter__(self):
    batches = iter(self.dl)
    for b in batches:
      yield (self.func(*b))


training_data = datasets.FashionMNIST(
  root='data',
  train=True,
  download=True,
  transform=ToTensor()
)
test_data = datasets.FashionMNIST(
  root='data',
  train=False,
  download=True,
  transform=ToTensor()
)

# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10 
lr = 0.1
loss_func = F.cross_entropy
model = NeuralNetwork().to(device)
print(model)
opt = optim.SGD(model.parameters(), lr=lr)
bs = 64

# data loader
train_dl = DataLoader(training_data, batch_size=bs, shuffle=True)
test_dl = DataLoader(test_data, batch_size=bs, shuffle=True)
# wrap data loader
train_dl = WrappedDataLoader(train_dl, preprocess)
test_dl = WrappedDataLoader(test_dl, preprocess)

fit(epochs, model, loss_func, opt, train_dl, test_dl)
