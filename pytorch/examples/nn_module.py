""" Use nn module for neural network """

import torch
import math

# create tensors to hold input and outputs
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# output y is a linear function of (x, x^2, x^3)
# hence it can be considered as a linear layer neural network
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# define model as a sequence of layers with the nn package
# layers receive a tensor as input, compute and output a tensor
# layers can also store weights an biases of that layer
model = torch.nn.Sequential(
  torch.nn.Linear(3, 1),
  torch.nn.Flatten(0, 1)
)

# nn contains popular loss functions
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
  # forward pass
  # model can be called like a function with input as model overrides the __call__ function
  y_pred = model(xx)

  # compute loss
  loss = loss_fn(y_pred, y)
  if t % 100 == 99:
    print(t, loss.item())

  # zero grads before backprop
  model.zero_grad()

  # backprop
  # compute gradient of the loss w.r.t. all learnable params of the model
  # learnable params -> tensors with requires_grad=True
  # internally the params of each module are stored in tensors with requires_grad=True
  loss.backward()

  # update weights using gradient descent
  with torch.no_grad():
    for param in model.parameters():
      param -= learning_rate * param.grad


# accessing the first layer of a model
linear_layer = model[0]

# for the linear layer, its params are stored as 'weight' and 'bias'
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + {linear_layer.weight[:,1].item()} x^2 + {linear_layer.weight[:,2].item()} x^3')
