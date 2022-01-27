""" Defining own autograd functions """

import torch
import math

class LegendrePolynomial3(torch.autograd.Function):
  """
  Own autograd functions can be implemented by sublassing torch.autograd.Function
  and implementing the forward and backward passes which operate on tensors.
  """
  @staticmethod
  def forward(ctx, input):
    """
    Forward pass gets a input tensor and computes the output tensor.
    ctx is a context object to stash information of backward computation.
    ctx is used for caching arbitrary objects for use in the backward pass.
    """
    ctx.save_for_backward(input)
    return 0.5 * (5*input**3 - 3*input)

  @staticmethod
  def backward(ctx, grad_ouput):
    """
    Backward pass receives a tensor containing the gradient of the loss w.r.t. the ouput,
    and computes the gradient of the loss w.r.t. the input.
    """
    input, = ctx.saved_tensors
    return grad_ouput*1.5 * (5*input**2 - 1)


dtype = torch.float
device = torch.device('cpu')

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
  # alias function for application as P3
  P3 = LegendrePolynomial3.apply

  # forward pass: compute predicted y using operations
  # compute P3 using custom autograd operation
  y_pred = a + b * P3(c + d*x)

  # compute loss
  loss = (y_pred - y).pow(2).sum()
  if t % 100 == 99:
    print(t, loss.item())

  # use autograd to compute the backward pass
  loss.backward()

  # update weights using gradient descent
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

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')
