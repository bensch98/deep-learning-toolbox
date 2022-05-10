import torch
import torch.nn as nn

model = nn.Sequential(
  nn.Linear(2, 8),
  nn.ReLU(),
  nn.Linear(8, 1)
).to('cpu')

state_dict = torch.load('mymodel.pth')
model.load_state_dict(state_dict)
model.to('cuda')
val_x = [[10, 11]]
val_x = torch.tensor(val_x).float().to('cuda')
model(val_x)

_y = model(val_x)
print(_y)

