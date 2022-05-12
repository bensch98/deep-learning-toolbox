import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.optim import SGD, Adam

from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary


def get_model():
  model = nn.Sequential(
              nn.Conv2d(1,1,kernel_size=3),
              nn.MaxPool2d(2),
              nn.ReLU(),
              nn.Flatten(),
              nn.Linear(1,1),
              nn.Sigmoid(),).to(device)
  loss_fn = nn.BCELoss()
  optimizer = Adam(model.parameters(), lr=1e-3)
  return model, loss_fn, optimizer


def train_batch(x, y, model, opt, loss_fn):
  model.train()
  prediction = model(x)
  batch_loss = loss_fn(prediction.squeeze(0),y)
  batch_loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  return batch_loss.item()


def analyze(model):
  (cnn_w, cnn_b), (lin_w, lin_b) = [(layer.weight.data, layer.bias.data) for layer in \
                                    list(model.children()) if hasattr(layer, 'weight')]
  h_im, w_im = X_train.shape[2:]
  h_conv, w_conv = cnn_w.shape[2:]
  sumprod = torch.zeros((h_im-h_conv+1, w_im-w_conv+1))

  for i in range(h_im-h_conv+1):
    for j in range(w_im-w_conv+1):
      img_subset = X_train[0, 0, i:(i+3), j:(j+3)]
      model_filter = cnn_w.reshape(3,3)
      val = torch.sum(img_subset*model_filter)+cnn_b
      sumprod[i,j] = val
      sumprod.clamp_min_(0)
      pooling_layer_output = torch.max(sumprod)
      intermediate_output_value = pooling_layer_output*lin_w+lin_b
  print(torch.sigmoid(intermediate_output_value))


if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  X_train = torch.tensor([[[[1,2,3,4],[2,3,4,5],[5,6,7,8],[1,3,4,5]]],
                          [[[-1,2,3,-4],[2,-3,4,5],[-5,6,-7,8],[-1,-3,-4,-5]]]]).to(device).float()
  X_train /= 8
  y_train = torch.tensor([0,1]).to(device).float()

  model, loss_fn, optimizer = get_model()
  summary(model, X_train)

  trn_dl = DataLoader(TensorDataset(X_train, y_train))

  for epoch in range(2000):
    for idx, batch in enumerate(iter(trn_dl)):
      x, y = batch
      batch_loss = train_batch(x, y, model, optimizer, loss_fn)

  _y = model(X_train[:1])
  print(_y)
  

  analyze(model)
