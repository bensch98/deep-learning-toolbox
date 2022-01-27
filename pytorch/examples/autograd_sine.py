""" Sine approximation with torch autograd """

import torch
import math

dtype = torch.float
# use cpu as transfer to GPU is too expensive for this case
device = torch.device('cpu')

# by default requires_grad=False
# this is ok for this case
# gradients are only needed for tensors which need to be backpropagated
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# create random tensors for weights
# requires_grad=True indicates that gradient computation w.r.t these tensors during backprop
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
  # forward pass: compute predicted y using operations on tensors
  y_pred = a + b*x + c*x**2 + d*x**3

  # compute loss
  # loss is then tensor of shape (1,)
  # loss.item() gets the scalar value held in tensor holding just one value
  loss = (y_pred - y).pow(2).sum()
  if t % 100 == 99:
    print(t, loss.item())

  # use autograd for backprop
  # this calculates all gradients of tensors with requires_grad=True 
  loss.backward()

  # manually update weights
  # torch.no_grad() disables gradient calculation for tensors with requires_grad=True
  # torch.no_grad() disables including these calculations in the computation graph
  with torch.no_grad():
    a -= learning_rate * a.grad
    b -= learning_rate * b.grad
    c -= learning_rate * c.grad
    d -= learning_rate * d.grad

    # manually zero the gradients after updating weights
    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
