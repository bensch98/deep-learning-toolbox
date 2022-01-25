import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

from basics import NeuralNetwork


if __name__ == '__main__':
  model = NeuralNetwork()
  model.load_state_dict(torch.load("model.pth"))

  classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
  ]

  test_data = datasets.FashionMNIST(
    root='../data',
    train=False,
    download=True,
    transform=ToTensor()
  )

  model.eval()
  for i in range(20):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
      pred = model(x)
      predicted, actual = classes[pred[0].argmax(0)], classes[y]
      print(f'Predicted: "{predicted}", Actual: "{actual}"')
