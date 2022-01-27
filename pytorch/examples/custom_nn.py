""" Implementing custom nn module """

import torch
import math

class Polynomial3(torch.nn.Module):
  def __init__(self):
    """
    The constructor instantiates here four params and assign them as member params.
    """
    super().__init__()
    self.a = torch.nn.Parameter(torch.randn(()))
    self.b = torch.nn.Parameter(torch.randn(()))
    self.c = torch.nn.Parameter(torch.randn(()))
    self.d = torch.nn.Parameter(torch.randn(()))
  

  def forward(self, x):
    """
    In the forward pass we accept a tensor of input data and must return a tensor of output data.
    Modules defined in the constructor as well as arbitrary operators on tensors are valid.
    """
    return self.a + self.b*x + self.c*x**2 + self.d*x**3

  def string(self):
    """
    Just like any class in Python, other custom methods can be defined on PyTorch modules.
    """
    return f'y = {self.a.item()} + {self.b.item()} * x + {self.c.item()} * x^2 + {self.d.item()} * x^3'


# create tensors to hold input and ouputs
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# construct the model by instantiating the defined class
model = Polynomial3()

# contruct loss function and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
  # forward pass: compute predicted y by passing x to the model
  y_pred = model(x)

  # compute loss
  loss = criterion(y_pred, y)
  if t % 100 == 99:
    print(t, loss.item())

  # zero grads
  optimizer.zero_grad()
  # backprop
  loss.backward()
  # update weights
  optimizer.step()

print(f'Result: {model.string()}')
