import torch
import torch.nn as nn
from torchvision import datasets

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ann_fmnist import MyNeuralNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
data_folder  = '../../../data'
fmnist = datasets.FashionMNIST(data_folder,
                               download=True,
                               train=True)

torch.manual_seed(420)
tr_images = fmnist.data
tr_targets = fmnist.targets
idx = np.random.randint(len(tr_images))
idx = 24300

plt.imshow(tr_images[idx], cmap='gray')
plt.title(fmnist.classes[tr_targets[idx]])
plt.show()

img = tr_images[idx]/255
img = img.view(28*28)
img = img.to(device)

model = MyNeuralNet().to(device)
#model = nn.Sequential(
#  nn.Dropout(0.25),
#  nn.Linear(28*28, 1000),
#  nn.ReLU(),
#  nn.Dropout(0.25),
#  nn.Linear(1000, 10),
#).to('cpu')
_y = np_output = model(img).cpu().detach().numpy()
tensor = np.exp(_y)/np.sum(np.exp(_y))

print(tensor)


state_dict = torch.load('../models/fmnist.pth')
model.load_state_dict(state_dict)
model.to('cpu')


preds = []
for px in range(-5,6):
  img = tr_images[idx]/255
  img = img.view(28,28)
  img2 = np.roll(img, px, axis=1)
  img3 = torch.Tensor(img2).view(28*28).to(device)
  _y = model(img3).cpu().detach().numpy()
  preds.append(np.exp(_y)/np.sum(np.exp(_y)))

fig, ax = plt.subplots(1,1,figsize=(12,10))
plt.title('Probability of each class for various translations')
sns.heatmap(np.array(preds), annot=True, ax=ax, fmt='.2f', xticklabels=fmnist.classes,
            yticklabels=[str(i)+str(' pixels') for i in range(-5,6)], cmap='gray')
plt.show()
