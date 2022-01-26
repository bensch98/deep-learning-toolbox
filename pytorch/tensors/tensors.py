import torch
import numpy as np


if __name__ == '__main__':
  # create tensor from python list object
  data = [[1, 2],
          [3, 4],
          [5, 6]]
  x_data = torch.tensor(data)

  # create tensor from numpy array
  np_array = np.array(data)
  x_np = torch.from_numpy(np_array)

  # create tensor with same properties as another tensor
  x_ones = torch.ones_like(x_data)
  print(f'Ones Tensor: \n {x_ones} \n')
  x_rand = torch.rand_like(x_data, dtype=torch.float)
  print(f'Random Tensor: \n {x_rand} \n')

  # create tensor with same shape as another tensor
  shape = (2, 3,)
  rand_tensor = torch.rand(shape)
  ones_tensor = torch.ones(shape)
  zeros_tensor = torch.zeros(shape)
  print(f'Random Tensor: \n {rand_tensor} \n')
  print(f'Ones Tensor: \n {ones_tensor} \n')
  print(f'Zeros Tensor: \n {zeros_tensor} \n')
  
  # tensor attributes
  tensor = torch.rand(3,4)
  print(f'Shape of tensor: {tensor.shape}')
  print(f'Datatype of tensor: {tensor.dtype}')
  print(f'Device tensor is stored on: {tensor.device}')

  '''
  # moving large tensors across different devices can be
  # computationally expensive in terms of time and memory
  if torch.cuda.is_available():
    # moving tensor to GPU has to be specified explicitly
    tensor = tensor.to('cuda')
  '''

  # tensor operations
  tensor = torch.ones(4, 4)
  print('First row: ', tensor[0])
  print('First column: ', tensor[:, 0])
  print('Last column: ', tensor[:, -1])
  tensor[:,1] = 0
  print(tensor)

  t1 = torch.cat([tensor, tensor, tensor], dim=1)
  print(t1)

  # matrix multiplication
  y1 = tensor @ tensor.T
  y2 = tensor.matmul(tensor.T)
  y3 = torch.rand_like(tensor)
  torch.matmul(tensor, tensor.T, out=y3)
  print(f'Tensor y1: \n {y1} \n')
  print(f'Tensor y2: \n {y2} \n')
  print(f'Tensor y3: \n {y3} \n')
  
  # element-wise multiplication
  z1 = tensor * tensor
  z2 = tensor.mul(tensor)
  z3 = torch.rand_like(tensor)
  torch.mul(tensor, tensor, out=z3)
  print(f'Tensor z1: \n {z1} \n')
  print(f'Tensor z2: \n {z2} \n')
  print(f'Tensor z3: \n {z3} \n')

  # converting one-element vector to numerical value
  agg = tensor.sum()
  agg_item = agg.item()
  print(agg_item, type(agg_item))

  # in-place operations
  print(tensor, '\n')
  tensor.add_(5)
  print(tensor)

  # bridge with numpy -> sharing of underlying memory
  t = torch.ones(5)
  print(f't: {t}')
  n = t.numpy()
  print(f'n: {n}')
  t.add_(1)
  print(f't: {t}')
  print(f'n: {n}')

  # sharing works in both directions
  data = [1, 2, 3, 4]
  n = np.array(data)
  print(f'n: {n}')
  t = torch.from_numpy(n)
  print(f't: {t}')
  t.add_(1)
  print(f'n: {n}')
  print(f't: {t}')
