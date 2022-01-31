import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class ActivationPlot:
  def __init__(self, x, func, title):
    """ Init activation function.
    :param x: Input x of activation function f(x).
    :param func: Activation function f(x).
    """
    self.x = x
    self.func = func
    self.title = title

    # save figure and plot globally
    self.fig = None
    self.plot = None

  def create_plot(self):
    """ Plots the activation function.
    :param title: Title of the plot.
    :param save: If yes, saves the plot, instead of showing it.
    """

    self.fig = plt.figure(figsize=(10,5))
    plt.plot(self.x, self.func)
    plt.axis('tight')
    plt.title(f'Activation Function: {self.title}')
    self.plot = plt

    return self

def save_plots(plts):
  # save activation functions
  for p in plts:
    filename = p.title.lower().replace(' ', '_')
    path = os.path.relpath(f'./img/{filename}')
    # defaults to png image
    p.fig.savefig(path, bbox_inches='tight', dpi=150)

def show_plots(plts):
  # show activation functions
  for p in plts:
    p.plot.show()

if __name__ == '__main__':
  act_funcs = []

  # same input x for all activation functions
  x = np.linspace(-10, 10)

  # binary step activation function
  y = np.heaviside(x, 1)
  binary_step = ActivationPlot(x, y, 'Binary Step').create_plot()
  act_funcs.append(binary_step)
  
  # linear activation fucntion
  y = x
  linear = ActivationPlot(x, y, 'Linear').create_plot()
  act_funcs.append(linear)
  
  # sigmoid activation function
  y = 1/(1+np.exp(-x))
  sigmoid = ActivationPlot(x, y, 'Sigmoid').create_plot() 
  act_funcs.append(sigmoid)

  # tanh activation function
  y = np.tanh(x)
  tanh = ActivationPlot(x, y, 'Tanh').create_plot()
  act_funcs.append(tanh)
  
  # rectified linear unit (ReLU)
  y = np.maximum(0, x)
  relu = ActivationPlot(x, y, 'ReLU').create_plot()
  act_funcs.append(relu)

  # softmax activation function
  y = np.exp(x)/np.sum(np.exp(x), axis=0)
  softmax = ActivationPlot(x, y, 'Softmax').create_plot()
  act_funcs.append(softmax)
  
  arg1 = sys.argv[1]
  if arg1 == 'show':
    show_plots(act_funcs)
  elif arg1 == 'save':
    save_plots(act_funcs)
  else:
    print('First argument must either be "show" or "save"')
