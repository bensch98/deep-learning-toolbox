import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from torchsummary import summary

import matplotlib.pyplot as plt

import time
import sys


# basic setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [[3], [7], [11], [15]]

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

X = X.to(device)
Y = Y.to(device)


class MyDataset(Dataset):
  def __init__(self, x, y):
    self.x = x.clone().detach().requires_grad_(True)
    self.y = y.clone().detach().requires_grad_(True)

  def __len__(self):
    return len(self.x)
  
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


class MyNeuralNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.input_to_hidden_layer = nn.Linear(2, 8)
    self.hidden_layer_activation = nn.ReLU()
    self.hidden_to_output_layer = nn.Linear(8, 1)

  def forward(self, x):
    x = self.input_to_hidden_layer(x)
    x = self.hidden_layer_activation(x)
    x = self.hidden_to_output_layer(x)
    return x


def my_mean_squared_error(_y, y):
  loss = (_y-y)**2
  loss = loss.mean()
  return loss


torch.manual_seed(420)

ds = MyDataset(X, Y)
dl = DataLoader(ds, batch_size=2, shuffle=True)

model = nn.Sequential(
  nn.Linear(2, 8),
  nn.ReLU(),
  nn.Linear(8, 1)).to(device)

#model = MyNeuralNet().to(device)

loss_func = nn.MSELoss()
#loss_func = my_mean_squared_error # same as the above
opt = SGD(model.parameters(), lr=0.001)

start = time.time()
loss_history = []
for _ in range(50):
  for x, y in dl:
    opt.zero_grad()
    loss_value = loss_func(model(x), y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value)

end = time.time() - start
print(end)
loss_history = [loss.detach().cpu().numpy() for loss in loss_history]

val_x = [[10, 11]]
val_x = torch.tensor(val_x).float().to(device)
_y = model(val_x)
print(f'Prediction:\n', _y)

summary(model, torch.zeros(1,2))

# save model with CPU tensors
torch.save(model.to('cpu').state_dict(), 'mymodel.pth')


plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()
