""" Use optim module for neural network """

import torch
import math

# create tensors to hold input and outputs
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# define model as a sequence of layers with the nn package
model = torch.nn.Sequential(
  torch.nn.Linear(3, 1),
  torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
# optimizer from optim package updates the weights automatically
# the first argument tells RMSprop which tensor will be updated
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
  # forward pass
  y_pred = model(xx)

  # compute loss
  loss = loss_fn(y_pred, y)
  if t % 100 == 99:
    print(t, loss.item())

  # zero grads before backprop
  # when using a optimizer from torch.optim, the gradients are reset to zero via the optimizer
  # zero_grad() is needed as the gradients are accumulated
  optimizer.zero_grad()

  # backprop: compute gradient of the loss w.r.t the model params
  loss.backward()
  
  # calling the step() function on an optimizer makes an update to its params
  optimizer.step()

# accessing the first layer of a model
linear_layer = model[0]

# for the linear layer, its params are stored as 'weight' and 'bias'
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + {linear_layer.weight[:,1].item()} x^2 + {linear_layer.weight[:,2].item()} x^3')
