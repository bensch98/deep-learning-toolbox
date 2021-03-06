""" Sine approximation with torch """

import torch
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sine_approx(device, iterations, learning_rate):
  start = time.time()

  dtype = torch.float

  # create random input and output data
  x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
  y = torch.sin(x)

  # randomly initialize weights
  a = torch.randn((), device=device, dtype=dtype)
  b = torch.randn((), device=device, dtype=dtype)
  c = torch.randn((), device=device, dtype=dtype)
  d = torch.randn((), device=device, dtype=dtype)

  for t in range(iterations):
    # forward pass: compute predicted y
    y_pred = a + b*x + c*x**2 + d*x**3

    # compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
      print(t, loss)

    # backprop to compute gradients of a, b, c, d w.r.t. loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred*x).sum()
    grad_c = (grad_y_pred*x**2).sum()
    grad_d = (grad_y_pred*x**3).sum()

    # update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
   
  print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
  return time.time() - start


if __name__ == '__main__':
  # compare GPU and CPU execution runtime
  duration_gpu = sine_approx('cuda', 100000, 1e-6)
  duration_cpu = sine_approx('cpu', 100000, 1e-6)
  print(f'Duration: {duration_gpu} seconds on cuda')
  print(f'Duration: {duration_cpu} seconds on cpu')

  # -> GPU is generally faster than CPU
  # -> if the cost of transferring tensors to the GPU is higher than the benefit CPU is faster
  # -> for very little calculation and small batch sizes CPU might be faster
